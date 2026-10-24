
import os
import requests
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SCRAPFLY_API_KEY")

if not API_KEY:
    raise ValueError("Missing SCRAPFLY_API_KEY in .env")

SEARCH_URL = "https://www.cars.com/shopping/results/"

# Test one selected model first
filters = {
    "stock_type": "used",
    "makes[]": "nissan",
    "models[]": "nissan-altima",
    "zip": "19136",
    "maximum_distance": "all",
    "page": 1
}

target_url = SEARCH_URL + "?" + urlencode(filters)

print("Target URL:", target_url)

response = requests.get(
    "https://api.scrapfly.io/scrape",
    params={
        "key": API_KEY,
        "url": target_url,
        "render_js": "true",
        "unblocker": "true",
        "country": "us"
    },
    timeout=120
)

response.raise_for_status()

data = response.json()
html = data.get("result", {}).get("content", "")

if not html:
    raise RuntimeError("No HTML returned from ScrapFly")

with open("cars_filtered_test.html", "w", encoding="utf-8") as f:
    f.write(html)

soup = BeautifulSoup(html, "html.parser")

cards = soup.select("fuse-card")

print("\nHTTP status:", response.status_code)
print("HTML characters:", len(html))
print("Listing cards found:", len(cards))

print("\nFirst 10 listing titles:")

for card in cards[:10]:
    heading = card.find("h2")

    if heading:
        print("-", heading.get_text(" ", strip=True))

print("\nSaved: cars_filtered_test.html")
