
import os
import re
import json
import time
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

OUTPUT_FILE = DATA_DIR / "honda_cars_collected.csv"
PROGRESS_FILE = DATA_DIR / "honda_cars_progress.json"

TARGETS = {
    "Civic": "honda-civic",
    "Accord": "honda-accord",
    "CR-V": "honda-cr_v"
}

MIN_YEAR = 2019
MAX_YEAR = 2024
MAX_PAGES_PER_MODEL = 10
DELAY_SECONDS = 3


def load_progress():
    if PROGRESS_FILE.exists():
        return set(json.loads(
            PROGRESS_FILE.read_text(encoding="utf-8")
        ))
    return set()


def save_progress(completed):
    temp = PROGRESS_FILE.with_suffix(".tmp")
    temp.write_text(
        json.dumps(sorted(completed), indent=2),
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


def parse_page(html, model):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    vehicle_links = 0

    for card in soup.select("fuse-card"):
        link = card.select_one('a[href*="/vehicledetail/"]')
        if not link:
            continue

        vehicle_links += 1
        heading = card.find("h2")
        title = (
            heading.get_text(" ", strip=True)
            if heading else link.get_text(" ", strip=True)
        )

        match = re.search(
            r"\b(20\d{2})\s+Honda\s+(.+)",
            title,
            re.IGNORECASE
        )
        if not match:
            continue

        year = int(match.group(1))
        remainder = match.group(2).strip()

        if not MIN_YEAR <= year <= MAX_YEAR:
            continue

        model_pattern = (
            r"CR[\s-]?V" if model == "CR-V"
            else re.escape(model)
        )

        model_match = re.match(
            rf"{model_pattern}\b",
            remainder,
            re.IGNORECASE
        )
        if not model_match:
            continue

        trim = remainder[model_match.end():].strip()
        text = card.get_text(" ", strip=True)

        price_match = re.search(r"\$([\d,]+)", text)
        mileage_match = re.search(
            r"([\d,]+)\s*mi\.?",
            text,
            re.IGNORECASE
        )

        if not price_match or not mileage_match:
            continue

        price = int(price_match.group(1).replace(",", ""))
        mileage = int(mileage_match.group(1).replace(",", ""))

        if price <= 0 or mileage < 0:
            continue

        listing_url = urljoin(
            "https://www.cars.com",
            link.get("href", "")
        ).split("?")[0]

        records.append({
            "source": "Cars.com",
            "year": year,
            "make": "Honda",
            "model": model,
            "trim": trim,
            "price": price,
            "mileage": mileage,
            "listing_url": listing_url,
            "description": text
        })

    return records, vehicle_links


def save_records(new_records):
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
    else:
        existing = pd.DataFrame()

    if new_records:
        combined = pd.concat(
            [existing, pd.DataFrame(new_records)],
            ignore_index=True
        )
    else:
        combined = existing

    if combined.empty:
        return 0, 0

    old_count = len(existing)
    combined = combined.drop_duplicates(
        subset=["listing_url"], keep="last"
    )

    temp = OUTPUT_FILE.with_suffix(".tmp")
    combined.to_csv(temp, index=False)
    temp.replace(OUTPUT_FILE)

    return len(combined), max(0, len(combined) - old_count)


def main():
    completed = load_progress()

    print("=" * 60)
    print("HONDA CARS.COM COLLECTOR")
    print("=" * 60)

    for model, slug in TARGETS.items():
        print(f"\nCollecting Honda {model}")

        for page in range(1, MAX_PAGES_PER_MODEL + 1):
            page_id = f"Honda|{model}|{page}"

            if page_id in completed:
                print(f"Page {page}: already completed")
                continue

            filters = {
                "stock_type": "used",
                "makes[]": "honda",
                "models[]": slug,
                "year_min": MIN_YEAR,
                "year_max": MAX_YEAR,
                "zip": "19136",
                "maximum_distance": "all",
                "page": page
            }

            url = (
                "https://www.cars.com/shopping/results/?"
                + urlencode(filters)
            )

            try:
                html = fetch_html(url)
                records, links = parse_page(html, model)

                if links == 0:
                    print("No vehicle links; stopping model.")
                    break

                if not records:
                    print(
                        f"Page {page}: {links} links but no "
                        "valid Honda records. Stopping for review."
                    )
                    break

                total, added = save_records(records)

                completed.add(page_id)
                save_progress(completed)

                print(
                    f"Page {page}: {len(records)} parsed, "
                    f"{added} new, {total} saved"
                )

            except Exception as error:
                print(f"Stopped safely: {error}")
                return

            time.sleep(DELAY_SECONDS)

    print("\nCollection finished.")

    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        print("\nTotal Honda Cars.com listings:", len(df))
        print(df.groupby("model").size().to_string())


if __name__ == "__main__":
    main()
