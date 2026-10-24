
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

API_KEY = os.getenv("SCRAPFLY_API_KEY")

if not API_KEY:
    raise ValueError("Missing SCRAPFLY_API_KEY")

API_URL = "https://api.scrapfly.io/scrape"

websites = {
    "Cars.com": "https://www.cars.com/shopping/results/?stock_type=used",
    "Carvana": "https://www.carvana.com/cars"
}

for name, url in websites.items():
    print(f"\nTesting {name}...")

    try:
        response = requests.get(
            API_URL,
            params={
                "key": API_KEY,
                "url": url,
                "render_js": "true",
                "unblocker": "true",
                "country": "us"
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()
        html = result.get("result", {}).get("content", "")

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else "None"

        print("HTTP status:", response.status_code)
        print("Page title:", title)
        print("HTML characters:", len(html))

        filename = name.lower().replace(".", "_") + "_test.html"

        with open(filename, "w", encoding="utf-8") as file:
            file.write(html)

        if "security verification" in html.lower() or \
           "performing security verification" in html.lower():
            print("WARNING: Security verification page returned.")
        else:
            print("Check saved HTML for vehicle listings.")

        print("Saved:", filename)

    except requests.RequestException as error:
        print("Request failed:", error)

print("\nTesting completed.")
