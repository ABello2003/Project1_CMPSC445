
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

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("SCRAPFLY_API_KEY")

if not API_KEY:
    raise ValueError("Missing SCRAPFLY_API_KEY in .env")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "cars_collected.csv"
PROGRESS_FILE = DATA_DIR / "cars_progress.json"

MAX_PAGES_PER_MODEL = 5
DELAY_SECONDS = 3

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
        "3 Series": "bmw-3_series",
        "5 Series": "bmw-5_series",
        "X3": "bmw-x3"
    }
}

BMW_PATTERNS = {
    "3 Series": r"\b(?:3 Series|3[234]\di|M340i)\b",
    "5 Series": r"\b(?:5 Series|5[234]\di|M550i)\b",
    "X3": r"\bX3\b"
}

# Pages confirmed successful in your previous run.
# These are used ONLY when no progress file exists.
PREVIOUSLY_COMPLETED = [
    "Nissan|Altima|1",
    "Nissan|Altima|2",
    "Nissan|Altima|3",
    "Nissan|Altima|4",
    "Nissan|Altima|5",
    "Nissan|Sentra|1",
    "Nissan|Sentra|2",
    "Nissan|Sentra|3",
    "Nissan|Sentra|4",
    "Nissan|Sentra|5",
    "Nissan|Rogue|1"
]


# ============================================================
# PROGRESS TRACKING
# ============================================================

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))

    # Only seed old progress when the old CSV is present.
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)

        if len(df) >= 216:
            print("Existing dataset found.")
            print("Restoring progress from previous run.")
            return set(PREVIOUSLY_COMPLETED)

        print("Existing CSV has fewer than 216 records.")
        print("Not assuming previous pages were completed.")

    return set()


def save_progress(completed):
    temporary_file = PROGRESS_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as f:
        json.dump(sorted(completed), f, indent=2)

    temporary_file.replace(PROGRESS_FILE)


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
            f"HTTP_ERROR: ScrapFly returned "
            f"status {response.status_code}"
        )

    try:
        result = response.json()
    except ValueError:
        raise RuntimeError(
            "INVALID_JSON: ScrapFly response was not valid JSON."
        ) from None

    html = result.get("result", {}).get("content", "")

    if not html:
        raise RuntimeError(
            "EMPTY_HTML: ScrapFly returned no page content."
        )

    return html


# ============================================================
# VEHICLE EXTRACTION
# ============================================================

def parse_page(html, make, model):
    soup = BeautifulSoup(html, "html.parser")

    records = []
    vehicle_links_seen = 0

    for card in soup.select("fuse-card"):

        link = card.select_one(
            'a[href*="/vehicledetail/"]'
        )

        if not link:
            continue

        vehicle_links_seen += 1

        heading = card.find("h2")

        title = (
            heading.get_text(" ", strip=True)
            if heading
            else link.get_text(" ", strip=True)
        )

        title_pattern = (
            rf"\b(19\d{{2}}|20\d{{2}})\s+"
            rf"{re.escape(make)}\s+(.+)"
        )

        match = re.search(
            title_pattern,
            title,
            re.IGNORECASE
        )

        if not match:
            continue

        year = int(match.group(1))
        remainder = match.group(2).strip()

        if make == "BMW":
            model_pattern = BMW_PATTERNS[model]
        else:
            model_pattern = rf"\b{re.escape(model)}\b"

        model_match = re.search(
            model_pattern,
            remainder,
            re.IGNORECASE
        )

        if not model_match:
            continue

        trim = remainder[model_match.end():].strip()

        text = card.get_text(" ", strip=True)

        price_match = re.search(
            r"\$([\d,]+)",
            text
        )

        mileage_match = re.search(
            r"([\d,]+)\s*mi\.",
            text,
            re.IGNORECASE
        )

        if not price_match or not mileage_match:
            continue

        price = int(
            price_match.group(1).replace(",", "")
        )

        mileage = int(
            mileage_match.group(1).replace(",", "")
        )

        listing_url = urljoin(
            "https://www.cars.com",
            link.get("href", "")
        ).split("?")[0]

        records.append({
            "source": "Cars.com",
            "year": year,
            "make": make,
            "model": model,
            "trim": trim,
            "price": price,
            "mileage": mileage,
            "listing_url": listing_url,
            "description": text
        })

    return records, vehicle_links_seen


# ============================================================
# SAVE VEHICLE RECORDS
# ============================================================

def save_records(new_records):
    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE)
    else:
        existing_df = pd.DataFrame()

    if new_records:
        new_df = pd.DataFrame(new_records)

        combined = pd.concat(
            [existing_df, new_df],
            ignore_index=True
        )
    else:
        combined = existing_df

    if combined.empty:
        return 0

    combined = combined.drop_duplicates(
        subset=["listing_url"],
        keep="last"
    )

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")
    combined.to_csv(temporary_file, index=False)
    temporary_file.replace(OUTPUT_FILE)

    return len(combined)


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary():
    if not OUTPUT_FILE.exists():
        print("No dataset saved yet.")
        return

    df = pd.read_csv(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("Total unique vehicles:", len(df))

    print("\nVehicles by make and model:")

    print(
        df.groupby(["make", "model"])
        .size()
        .to_string()
    )

    print("\nDataset saved to:", OUTPUT_FILE)


# ============================================================
# MAIN COLLECTION LOOP
# ============================================================

def main():
    completed = load_progress()

    total = (
        len(pd.read_csv(OUTPUT_FILE))
        if OUTPUT_FILE.exists()
        else 0
    )

    print("=" * 60)
    print("CARS.COM VEHICLE COLLECTOR")
    print("=" * 60)

    print("Previously saved vehicles:", total)
    print("Previously completed pages:", len(completed))
    print("Maximum pages per model:", MAX_PAGES_PER_MODEL)

    for make, models in TARGETS.items():

        for model, slug in models.items():

            print("\n" + "=" * 55)
            print(f"COLLECTING: {make} {model}")
            print("=" * 55)

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

                filters = {
                    "stock_type": "used",
                    "makes[]": make.lower(),
                    "models[]": slug,
                    "zip": "19136",
                    "maximum_distance": "all",
                    "page": page
                }

                target_url = (
                    "https://www.cars.com/shopping/results/?"
                    + urlencode(filters)
                )

                try:
                    html = fetch_html(target_url)

                    records, links = parse_page(
                        html,
                        make,
                        model
                    )

                    total = save_records(records)

                    # Save checkpoint after CSV is written.
                    completed.add(page_id)
                    save_progress(completed)

                    print(
                        f"Page {page}: "
                        f"{links} vehicle links, "
                        f"{len(records)} matching records, "
                        f"{total} unique saved"
                    )

                    if links == 0:
                        print(
                            "No vehicle links found. "
                            "Stopping this model."
                        )
                        break

                except RuntimeError as error:

                    if str(error) == "RATE_LIMIT":
                        print("\n" + "=" * 60)
                        print("SCRAPFLY RATE LIMIT REACHED")
                        print("=" * 60)

                        print(
                            "Stopping ALL requests immediately."
                        )
                        print(
                            "Check your ScrapFly account "
                            "usage and limits."
                        )
                        print(
                            "Your existing CSV and "
                            "progress have been preserved."
                        )

                        print_summary()
                        return

                    print(
                        f"Stopped on {make} {model}, "
                        f"page {page}: {error}"
                    )
                    print(
                        "No further requests will be made."
                    )

                    print_summary()
                    return

                except Exception:
                    print(
                        f"Unexpected error on "
                        f"{make} {model}, page {page}."
                    )
                    print(
                        "Stopping safely. "
                        "Progress is preserved."
                    )

                    print_summary()
                    return

                time.sleep(DELAY_SECONDS)

    print("\n" + "=" * 60)
    print("COLLECTION FINISHED")
    print("=" * 60)

    print_summary()


if __name__ == "__main__":
    main()
