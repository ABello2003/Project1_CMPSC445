
import os
import re
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SCRAPFLY_API_KEY")
if not API_KEY:
    raise ValueError("Missing SCRAPFLY_API_KEY in .env")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "honda_carvana_collected.csv"
PROGRESS_FILE = DATA_DIR / "honda_carvana_progress.json"

TARGETS = {
    "Civic": "honda-civic",
    "Accord": "honda-accord",
    "CR-V": "honda-cr-v"
}

MIN_YEAR = 2019
MAX_YEAR = 2024
MAX_PAGES_PER_MODEL = 10
DELAY_SECONDS = 3
MAX_STALE_PAGES = 2


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(
            PROGRESS_FILE.read_text(encoding="utf-8")
        )
    return {
        "completed": [],
        "page_signatures": {},
        "stopped_models": []
    }


def save_progress(progress):
    temp = PROGRESS_FILE.with_suffix(".tmp")
    temp.write_text(
        json.dumps(progress, indent=2),
        encoding="utf-8"
    )
    temp.replace(PROGRESS_FILE)


def fetch_html(url):
    try:
        response = requests.get(
            "https://api.scrapfly.io/scrape",
            params={
                "key": API_KEY,
                "url": url,
                "render_js": "true",
                "unblocker": "true",
                "country": "us"
            },
            timeout=120
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"Network request failed: {error}"
        ) from error

    if response.status_code == 429:
        raise RuntimeError("RATE_LIMIT")

    response.raise_for_status()
    result = response.json().get("result", {})

    status = result.get("status_code")
    if status and int(status) >= 400:
        raise RuntimeError(f"Target returned HTTP {status}")

    html = result.get("content", "")
    if not html:
        raise RuntimeError("EMPTY_HTML")

    return html


def normalize_model(make, model):
    if isinstance(make, dict):
        make = make.get("name", "")

    if isinstance(model, dict):
        model = model.get("name", "")

    if str(make).strip().lower() != "honda":
        return None

    value = str(model).strip().lower()

    if value == "civic":
        return "Civic"
    if value == "accord":
        return "Accord"
    if re.fullmatch(r"cr[\s-]?v", value):
        return "CR-V"

    return None


def extract_trim(item, year, model):
    description = str(item.get("description") or "")
    title = str(item.get("name") or "")

    model_pattern = (
        r"CR[\s-]?V" if model == "CR-V"
        else re.escape(model)
    )

    patterns = [
        (
            rf"(?:Used\s+)?{year}\s+Honda\s+"
            rf"{model_pattern}\s+(.+?)\s+with\s+"
            rf"[\d,]+\s+miles"
        ),
        (
            rf"^{year}\s+Honda\s+"
            rf"{model_pattern}\s+(.+)$"
        )
    ]

    for text in (description, title):
        for pattern in patterns:
            match = re.search(
                pattern, text, re.IGNORECASE
            )
            if match:
                trim = match.group(1).strip()
                trim = re.split(
                    r"\s+[-|]\s+|\s+\$\d",
                    trim
                )[0].strip()
                return trim

    return ""


def walk_json(value):
    if isinstance(value, list):
        for element in value:
            yield from walk_json(element)
    elif isinstance(value, dict):
        yield value
        for key in ("@graph", "itemListElement"):
            if key in value:
                yield from walk_json(value[key])
        if "item" in value and isinstance(
            value["item"], (dict, list)
        ):
            yield from walk_json(value["item"])


def extract_vehicles(html):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    vehicle_count = 0

    for script in soup.find_all(
        "script", type="application/ld+json"
    ):
        try:
            payload = json.loads(
                script.string or script.get_text()
            )
        except (ValueError, TypeError):
            continue

        for item in walk_json(payload):
            types = item.get("@type", [])
            if isinstance(types, str):
                types = [types]

            if not any(
                str(t).split("/")[-1] == "Vehicle"
                for t in types
            ):
                continue

            vehicle_count += 1

            make = item.get("manufacturer") or item.get("brand")
            model = normalize_model(make, item.get("model"))

            if not model:
                continue

            offers = item.get("offers", {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if not isinstance(offers, dict):
                offers = {}

            mileage = item.get("mileageFromOdometer")
            if isinstance(mileage, dict):
                mileage = mileage.get("value")

            try:
                year = int(item.get("modelDate"))
                price = float(
                    str(offers.get("price"))
                    .replace("$", "").replace(",", "")
                )
                miles = int(
                    float(str(mileage).replace(",", ""))
                )
            except (ValueError, TypeError):
                continue

            if not MIN_YEAR <= year <= MAX_YEAR:
                continue

            if price <= 0 or miles < 0:
                continue

            listing_url = offers.get("url")
            if not listing_url:
                continue

            listing_url = urljoin(
                "https://www.carvana.com",
                str(listing_url)
            ).split("?")[0]

            records.append({
                "source": "Carvana",
                "year": year,
                "make": "Honda",
                "model": model,
                "trim": extract_trim(item, year, model),
                "price": price,
                "mileage": miles,
                "color": item.get("color"),
                "condition": item.get("itemCondition"),
                "vin": item.get(
                    "vehicleIdentificationNumber"
                ),
                "sku": item.get("sku"),
                "title": item.get("name"),
                "description": item.get("description"),
                "listing_url": listing_url
            })

    return records, vehicle_count


def deduplicate(df):
    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=["listing_url"], keep="last"
    )

    if "vin" in df.columns:
        vin = df["vin"].fillna("").astype(str).str.strip()
        valid = vin.str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}")

        with_vin = df[valid].drop_duplicates(
            subset=["vin"], keep="last"
        )
        without_vin = df[~valid]

        df = pd.concat(
            [with_vin, without_vin],
            ignore_index=True
        )

    return df


def save_records(records):
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(
            OUTPUT_FILE, dtype={"vin": "string"}
        )
    else:
        existing = pd.DataFrame()

    old_count = len(existing)

    combined = pd.concat(
        [existing, pd.DataFrame(records)],
        ignore_index=True
    )

    combined = deduplicate(combined)

    temp = OUTPUT_FILE.with_suffix(".tmp")
    combined.to_csv(temp, index=False)
    temp.replace(OUTPUT_FILE)

    return len(combined), max(0, len(combined) - old_count)


def page_signature(records):
    urls = sorted(
        str(record["listing_url"])
        for record in records
    )
    return hashlib.sha256(
        "|".join(urls).encode("utf-8")
    ).hexdigest()


def main():
    progress = load_progress()
    completed = set(progress.get("completed", []))
    signatures = progress.get("page_signatures", {})
    stopped = set(progress.get("stopped_models", []))

    print("=" * 60)
    print("HONDA CARVANA COLLECTOR")
    print("=" * 60)

    for model, slug in TARGETS.items():
        if model in stopped:
            print(f"Skipping previously stopped model: {model}")
            continue

        stale_pages = 0
        print(f"\nCollecting Honda {model}")

        for page in range(1, MAX_PAGES_PER_MODEL + 1):
            page_id = f"Honda|{model}|{page}"

            if page_id in completed:
                print(f"Page {page}: already completed")
                continue

            url = (
                f"https://www.carvana.com/cars/{slug}?"
                + urlencode({"page": page})
            )

            try:
                html = fetch_html(url)
                records, found = extract_vehicles(html)

                matching = [
                    r for r in records
                    if r["model"] == model
                ]

                if found == 0:
                    print(
                        "No structured vehicle data found. "
                        "Stopping this model for inspection."
                    )
                    break

                if not matching:
                    print(
                        f"Page {page}: {found} vehicles found, "
                        "but none matched the requested model."
                    )
                    break

                signature = page_signature(matching)
                previous = {
                    value for key, value in signatures.items()
                    if key.startswith(f"Honda|{model}|")
                }

                if signature in previous:
                    print(
                        "Repeated page detected. "
                        "Stopping to save ScrapFly credits."
                    )
                    stopped.add(model)
                    break

                total, added = save_records(matching)

                completed.add(page_id)
                signatures[page_id] = signature

                print(
                    f"Page {page}: {len(matching)} matching, "
                    f"{added} new, {total} saved"
                )

                if added == 0:
                    stale_pages += 1
                else:
                    stale_pages = 0

                if stale_pages >= MAX_STALE_PAGES:
                    stopped.add(model)
                    print("Repeatedly no new cars; stopping model.")
                    break

            except Exception as error:
                print(f"Stopped safely: {error}")
                return

            finally:
                progress["completed"] = sorted(completed)
                progress["page_signatures"] = signatures
                progress["stopped_models"] = sorted(stopped)
                save_progress(progress)

            time.sleep(DELAY_SECONDS)

    print("\nCollection finished.")

    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        print("\nTotal Honda Carvana listings:", len(df))
        print(df.groupby("model").size().to_string())


if __name__ == "__main__":
    main()

