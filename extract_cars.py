
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

HTML_FILE = "cars_filtered_test.html"
OUTPUT_FILE = Path("data/cars_raw.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TARGET_MODELS = {
    "Nissan": ["Altima", "Sentra", "Rogue"],
    "BMW": ["3 Series", "5 Series", "X3"],
    "Toyota": ["Camry", "Corolla", "RAV4"]
}

BMW_ALIASES = {
    "3 Series": r"\b(?:3 Series|3[234]\di|M340i)\b",
    "5 Series": r"\b(?:5 Series|5[234]\di|M550i)\b",
    "X3": r"\bX3\b"
}


def identify_vehicle(title):
    match = re.search(
        r"\b(19\d{2}|20\d{2})\s+(Nissan|BMW|Toyota)\s+(.+)",
        title,
        re.IGNORECASE
    )

    if not match:
        return None

    year = int(match.group(1))
    make = match.group(2).upper() if match.group(2).lower() == "bmw" else match.group(2).title()
    remainder = match.group(3).strip()

    for model in TARGET_MODELS[make]:
        pattern = (
            BMW_ALIASES[model]
            if make == "BMW"
            else rf"\b{re.escape(model)}\b"
        )

        found = re.search(pattern, remainder, re.IGNORECASE)

        if found:
            return {
                "year": year,
                "make": make,
                "model": model,
                "trim": remainder[found.end():].strip()
            }

    return None


def extract_cars(html):
    soup = BeautifulSoup(html, "html.parser")
    vehicles = []

    cards = soup.select("fuse-card")

    print("Total fuse-card elements:", len(cards))

    vehicle_cards = 0

    for card in cards:
        link = card.select_one('a[href*="/vehicledetail/"]')

        if not link:
            continue

        vehicle_cards += 1

        title = link.get_text(" ", strip=True)

        if not re.search(r"\b(19\d{2}|20\d{2})\b", title):
            heading = card.find("h2")
            title = heading.get_text(" ", strip=True) if heading else ""

        vehicle = identify_vehicle(title)

        if not vehicle:
            continue

        text = card.get_text(" ", strip=True)

        price_match = re.search(r"\$([\d,]+)", text)
        mileage_match = re.search(
            r"([\d,]+)\s*mi\.",
            text,
            re.IGNORECASE
        )

        if not price_match or not mileage_match:
            continue

        price = int(price_match.group(1).replace(",", ""))
        mileage = int(mileage_match.group(1).replace(",", ""))

        url = urljoin(
            "https://www.cars.com",
            link.get("href", "")
        ).split("?")[0]

        vehicles.append({
            "source": "Cars.com",
            "year": vehicle["year"],
            "make": vehicle["make"],
            "model": vehicle["model"],
            "trim": vehicle["trim"],
            "price": price,
            "mileage": mileage,
            "listing_url": url,
            "description": text
        })

    print("Cards with vehicle links:", vehicle_cards)

    return pd.DataFrame(vehicles)


def main():
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        html = f.read()

    df = extract_cars(html)

    if df.empty:
        print("No matching vehicles extracted.")
        return

    df = df.drop_duplicates(subset=["listing_url"])

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEXTRACTION RESULTS")
    print("=" * 60)

    print("Matching vehicles extracted:", len(df))

    print("\nVehicle counts by make and model:")
    print(df.groupby(["make", "model"]).size())

    print("\nFirst 10 vehicles:")
    print(
        df[
            ["year", "make", "model", "trim", "price", "mileage"]
        ].head(10).to_string(index=False)
    )

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
