
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Extract structured vehicle records from Carvana
with open("carvana_test.html", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

vehicles = []

for script in soup.find_all("script", type="application/ld+json"):
    try:
        data = json.loads(script.string or script.get_text())
    except (json.JSONDecodeError, TypeError):
        continue

    records = data if isinstance(data, list) else [data]

    for record in records:
        if not isinstance(record, dict):
            continue

        if record.get("@type") != "Vehicle":
            continue

        offer = record.get("offers", {})
        if not isinstance(offer, dict):
            offer = {}

        description = record.get("description", "")

        vehicles.append({
            "source": "Carvana",
            "year": record.get("modelDate"),
            "make": record.get("manufacturer"),
            "model": record.get("model"),
            "price": offer.get("price"),
            "mileage": record.get("mileageFromOdometer"),
            "color": record.get("color"),
            "condition": record.get("itemCondition"),
            "vin": record.get("vehicleIdentificationNumber"),
            "description": description,
            "listing_url": offer.get("url"),
            "sku": record.get("sku")
        })

df = pd.DataFrame(vehicles)

if not df.empty:
    df = df.drop_duplicates(subset=["vin"])
    df.to_csv(DATA_DIR / "carvana_raw.csv", index=False)

print("\nCARVANA EXTRACTION")
print("Vehicles extracted:", len(df))
print(df[["year", "make", "model", "price", "mileage"]].head(10))

# Examine Cars.com HTML for vehicle listing cards
with open("cars_com_test.html", encoding="utf-8") as f:
    cars_soup = BeautifulSoup(f.read(), "html.parser")

cars_links = []

for link in cars_soup.find_all("a", href=True):
    href = link["href"]

    if "/vehicledetail/" in href:
        cars_links.append(href.split("?")[0])

cars_links = list(dict.fromkeys(cars_links))

print("\nCARS.COM INSPECTION")
print("Unique vehicle links:", len(cars_links))

for url in cars_links[:10]:
    print(url)

print("\nCompleted.")
