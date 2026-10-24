
import os
import re
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("SCRAPFLY_API_KEY")

if not API_KEY:
    raise ValueError("Missing SCRAPFLY_API_KEY in .env")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "carvana_collected.csv"

# New progress file because the previous search URLs were wrong.
PROGRESS_FILE = DATA_DIR / "carvana_progress_v2.json"

TARGET_COUNT = 500
MAX_PAGES_PER_MODEL = 10
DELAY_SECONDS = 3

# Stop a model if multiple consecutive pages add no new cars.
MAX_STALE_PAGES = 2

# Model-specific URL slugs.
# Toyota Camry has been verified; other slugs will be checked
# against the actual returned vehicles.
TARGETS = {
    "Nissan": {
        "Altima": "nissan-altima",
        "Sentra": "nissan-sentra",
        "Rogue": "nissan-rogue"
    },
    "Toyota": {
        "Camry": "toyota-camry",
        "Corolla": "toyota-corolla",
        "RAV4": "toyota-rav4"
    },
    "BMW": {
        "3 Series": "bmw-3-series",
        "5 Series": "bmw-5-series",
        "X3": "bmw-x3"
    }
}

# Normalize BMW model names.
BMW_PATTERNS = {
    "3 Series": r"^(?:3 Series|3[234]\di|M340i)$",
    "5 Series": r"^(?:5 Series|5[234]\di|M550i)$",
    "X3": r"^X3$"
}

# ============================================================
# PROGRESS TRACKING
# ============================================================

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "completed": [],
        "page_signatures": {},
        "stopped_models": []
    }


def save_progress(progress):
    temp = PROGRESS_FILE.with_suffix(".tmp")

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

    temp.replace(PROGRESS_FILE)


# ============================================================
# SCRAPFLY REQUEST
# ============================================================

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

    except requests.RequestException:
        raise RuntimeError(
            "NETWORK_ERROR: Unable to contact ScrapFly."
        ) from None

    if response.status_code == 429:
        raise RuntimeError("RATE_LIMIT")

    if response.status_code != 200:
        raise RuntimeError(
            f"SCRAPFLY_HTTP_{response.status_code}"
        )

    try:
        result = response.json()
    except ValueError:
        raise RuntimeError("INVALID_JSON") from None

    scrape_result = result.get("result", {})

    target_status = scrape_result.get("status_code")

    if target_status == 429:
        raise RuntimeError("RATE_LIMIT")

    if target_status and int(target_status) >= 400:
        raise RuntimeError(
            f"TARGET_HTTP_{target_status}"
        )

    html = scrape_result.get("content", "")

    if not html:
        raise RuntimeError("EMPTY_HTML")

    return html


# ============================================================
# MODEL NORMALIZATION
# ============================================================

def normalize_model(make, model):
    if not make or not model:
        return None

    make = str(make).strip().lower()
    model = str(model).strip()

    if make == "nissan":
        for target in TARGETS["Nissan"]:
            if model.lower() == target.lower():
                return "Nissan", target

    if make == "toyota":
        for target in TARGETS["Toyota"]:
            if model.lower() == target.lower():
                return "Toyota", target

    if make == "bmw":
        for target, pattern in BMW_PATTERNS.items():
            if re.fullmatch(
                pattern, model, re.IGNORECASE
            ):
                return "BMW", target

    return None


# ============================================================
# EXTRACT TRIM
# ============================================================

def extract_trim(description, year, make, model):
    if not description:
        return ""

    # Example:
    # Used 2025 Toyota Camry SE with 25000 miles - $25000
    pattern = (
        rf"Used\s+{year}\s+"
        rf"{re.escape(make)}\s+"
        rf"{re.escape(model)}\s*"
        rf"(.*?)\s+with\s+[\d,]+\s+miles"
    )

    match = re.search(
        pattern,
        description,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


# ============================================================
# EXTRACT VEHICLE DATA
# ============================================================

def extract_vehicles(html):
    soup = BeautifulSoup(html, "html.parser")

    records = []
    total_vehicle_records = 0

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    for script in scripts:
        try:
            data = json.loads(
                script.string or script.get_text()
            )
        except (ValueError, TypeError):
            continue

        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            item_type = item.get("@type", "")

            if isinstance(item_type, list):
                is_vehicle = "Vehicle" in item_type
            else:
                is_vehicle = item_type == "Vehicle"

            if not is_vehicle:
                continue

            total_vehicle_records += 1

            make = item.get("manufacturer") or item.get("brand")
            model = item.get("model")

            if isinstance(make, dict):
                make = make.get("name")

            if isinstance(model, dict):
                model = model.get("name")

            normalized = normalize_model(make, model)

            if normalized is None:
                continue

            make, model = normalized

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
                    .replace("$", "")
                    .replace(",", "")
                )
                mileage = int(
                    float(str(mileage).replace(",", ""))
                )
            except (TypeError, ValueError):
                continue

            if not 1990 <= year <= 2030:
                continue

            if price <= 0 or mileage < 0:
                continue

            listing_url = offers.get("url")

            if not listing_url:
                continue

            description = str(
                item.get("description") or ""
            )

            trim = extract_trim(
                description,
                year,
                make,
                model
            )

            records.append({
                "source": "Carvana",
                "year": year,
                "make": make,
                "model": model,
                "trim": trim,
                "price": price,
                "mileage": mileage,
                "color": item.get("color"),
                "condition": item.get("itemCondition"),
                "vin": item.get(
                    "vehicleIdentificationNumber"
                ),
                "sku": item.get("sku"),
                "title": item.get("name"),
                "description": description,
                "listing_url": listing_url.split("?")[0]
            })

    return records, total_vehicle_records


# ============================================================
# DEDUPLICATION AND SAVING
# ============================================================

def deduplicate(df):
    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=["listing_url"],
        keep="last"
    )

    if "vin" in df.columns:
        valid_vin = (
            df["vin"].notna()
            & df["vin"].astype(str).str.strip().ne("")
        )

        with_vin = df[valid_vin].drop_duplicates(
            subset=["vin"],
            keep="last"
        )

        without_vin = df[~valid_vin]

        df = pd.concat(
            [with_vin, without_vin],
            ignore_index=True
        )

    return df


def save_records(records):
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(
            OUTPUT_FILE,
            dtype={"vin": "string", "sku": "string"}
        )
    else:
        existing = pd.DataFrame()

    if records:
        new_df = pd.DataFrame(records)

        combined = pd.concat(
            [existing, new_df],
            ignore_index=True
        )
    else:
        combined = existing

    if combined.empty:
        return 0, 0

    old_count = len(deduplicate(existing))
    combined = deduplicate(combined)

    new_count = len(combined)
    added = new_count - old_count

    temp = OUTPUT_FILE.with_suffix(".tmp")
    combined.to_csv(temp, index=False)
    temp.replace(OUTPUT_FILE)

    return new_count, added


# ============================================================
# PAGINATION SIGNATURE
# ============================================================

def page_signature(records):
    """
    Detect identical listing sets returned for different pages.
    """
    identifiers = sorted(
        str(record["listing_url"])
        for record in records
    )

    content = "|".join(identifiers)

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# SUMMARY
# ============================================================

def print_summary():
    if not OUTPUT_FILE.exists():
        print("No Carvana data saved.")
        return

    df = pd.read_csv(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("CARVANA COLLECTION SUMMARY")
    print("=" * 60)

    print("Total unique listings:", len(df))

    print("\nListings by make and model:")

    print(
        df.groupby(["make", "model"])
        .size()
        .to_string()
    )

    print("\nSaved to:", OUTPUT_FILE)


# ============================================================
# MAIN COLLECTION LOOP
# ============================================================

def main():
    progress = load_progress()

    completed = set(
        progress.get("completed", [])
    )

    signatures = progress.get(
        "page_signatures", {}
    )

    stopped_models = set(
        progress.get("stopped_models", [])
    )

    total = (
        len(pd.read_csv(OUTPUT_FILE))
        if OUTPUT_FILE.exists()
        else 0
    )

    print("=" * 60)
    print("CARVANA VEHICLE COLLECTOR V2")
    print("=" * 60)

    print("Previously saved vehicles:", total)
    print("Completed pages:", len(completed))
    print("Target vehicles:", TARGET_COUNT)

    if total >= TARGET_COUNT:
        print("\nTarget already reached.")
        print_summary()
        return

    for make, models in TARGETS.items():
        for model, slug in models.items():

            model_id = f"{make}|{model}"

            if model_id in stopped_models:
                print(
                    f"\nSkipping {make} {model}: "
                    "previously stopped due to repeated results."
                )
                continue

            print("\n" + "=" * 55)
            print(f"COLLECTING: {make} {model}")
            print("=" * 55)

            stale_pages = 0

            for page in range(
                1,
                MAX_PAGES_PER_MODEL + 1
            ):
                page_id = f"{make}|{model}|{page}"

                if page_id in completed:
                    print(
                        f"Page {page}: "
                        "Already completed. Skipping."
                    )
                    continue

                params = {"page": page}

                target_url = (
                    f"https://www.carvana.com/cars/{slug}?"
                    + urlencode(params)
                )

                try:
                    html = fetch_html(target_url)

                    records, found = extract_vehicles(html)

                    matching = [
                        record for record in records
                        if record["make"] == make
                        and record["model"] == model
                    ]

                    # Do not mark invalid/empty pages as completed.
                    if found == 0:
                        print(
                            f"Page {page}: "
                            "No structured vehicle records found."
                        )
                        print(
                            "Stopping this model for inspection."
                        )
                        break

                    if not matching:
                        print(
                            f"Page {page}: "
                            f"{found} vehicles found, "
                            "0 matching requested model."
                        )
                        print(
                            "Search URL may not be valid."
                        )
                        break

                    signature = page_signature(matching)

                    previous_signatures = {
                        value
                        for key, value in signatures.items()
                        if key.startswith(model_id + "|")
                    }

                    if signature in previous_signatures:
                        print(
                            f"Page {page}: "
                            "Repeated vehicle page detected."
                        )
                        print(
                            "Stopping this model to avoid "
                            "wasting ScrapFly credits."
                        )

                        stopped_models.add(model_id)
                        break

                    total, added = save_records(matching)

                    print(
                        f"Page {page}: "
                        f"{found} vehicle records, "
                        f"{len(matching)} matching, "
                        f"{added} NEW, "
                        f"{total} total unique"
                    )

                    completed.add(page_id)
                    signatures[page_id] = signature

                    if added == 0:
                        stale_pages += 1
                    else:
                        stale_pages = 0

                    if stale_pages >= MAX_STALE_PAGES:
                        print(
                            "Multiple pages added no new vehicles."
                        )
                        print("Stopping this model.")

                        stopped_models.add(model_id)
                        break

                    if total >= TARGET_COUNT:
                        print("\nTARGET OF 500 REACHED!")
                        break

                except RuntimeError as error:
                    if str(error) == "RATE_LIMIT":
                        print("\nSCRAPFLY RATE LIMIT REACHED")
                        print("Stopping all requests.")
                    else:
                        print(
                            f"Collection stopped: {error}"
                        )

                    progress["completed"] = sorted(completed)
                    progress["page_signatures"] = signatures
                    progress["stopped_models"] = sorted(
                        stopped_models
                    )
                    save_progress(progress)

                    print_summary()
                    return

                except Exception:
                    print(
                        f"Unexpected error on "
                        f"{make} {model}, page {page}."
                    )
                    print("Stopping safely.")
                    print_summary()
                    return

                finally:
                    progress["completed"] = sorted(completed)
                    progress["page_signatures"] = signatures
                    progress["stopped_models"] = sorted(
                        stopped_models
                    )
                    save_progress(progress)

                if total >= TARGET_COUNT:
                    break

                time.sleep(DELAY_SECONDS)

            if total >= TARGET_COUNT:
                break

        if total >= TARGET_COUNT:
            break

    print("\n" + "=" * 60)
    print("COLLECTION FINISHED")
    print("=" * 60)

    print_summary()


if __name__ == "__main__":
    main()
