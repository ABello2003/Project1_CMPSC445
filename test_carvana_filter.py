
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("SCRAPFLY_API_KEY")

if not API_KEY:
    raise ValueError("Missing SCRAPFLY_API_KEY")

url = "https://www.carvana.com/cars/toyota-camry"

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

if response.status_code != 200:
    print("ScrapFly HTTP status:", response.status_code)
    raise SystemExit

data = response.json()
html = data.get("result", {}).get("content", "")

with open("carvana_filtered_test.html", "w", encoding="utf-8") as f:
    f.write(html)

soup = BeautifulSoup(html, "html.parser")

vehicles = []

for script in soup.find_all("script", type="application/ld+json"):
    try:
        item = json.loads(script.string or script.get_text())
    except (ValueError, TypeError):
        continue

    if not isinstance(item, dict):
        continue

    if item.get("@type") == "Vehicle":
        vehicles.append(item)

print("\nCARVANA FILTER TEST")
print("=" * 50)
print("HTML characters:", len(html))
print("Vehicles found:", len(vehicles))

camry_count = 0

for vehicle in vehicles:
    make = vehicle.get("manufacturer", "")
    model = vehicle.get("model", "")

    print(
        vehicle.get("modelDate"),
        make,
        model
    )

    if (
        str(make).lower() == "toyota"
        and str(model).lower() == "camry"
    ):
        camry_count += 1

print("\nToyota Camry matches:", camry_count)
print("Saved HTML: carvana_filtered_test.html")
