
import json
from bs4 import BeautifulSoup

files = [
    "cars_com_test.html",
    "carvana_test.html"
]

for filename in files:
    print("\n" + "=" * 60)
    print("INSPECTING:", filename)
    print("=" * 60)

    with open(filename, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Check structured JSON-LD data
    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    print("\nJSON-LD scripts found:", len(scripts))

    for i, script in enumerate(scripts[:5]):
        try:
            data = json.loads(script.string or script.get_text())
            print(f"\nJSON-LD #{i + 1}:")
            print(json.dumps(data, indent=2)[:1500])
        except (ValueError, TypeError):
            continue

    # Look for common vehicle listing elements
    selectors = [
        "[data-testid*='vehicle']",
        "[data-test*='vehicle']",
        "[class*='vehicle-card']",
        "[class*='vehicleCard']",
        "[class*='inventory-card']",
        "[class*='listing-row']",
        "[class*='vehicle-tile']"
    ]

    print("\nPossible listing elements:")

    for selector in selectors:
        elements = soup.select(selector)
        print(f"{selector}: {len(elements)}")

        if elements:
            print(
                "Example:",
                elements[0].get_text(" ", strip=True)[:350]
            )

    # Look for links to individual vehicles
    links = soup.find_all("a", href=True)

    vehicle_links = []

    for link in links:
        href = link.get("href", "")

        if (
            "/vehicledetail/" in href
            or "/vehicle/" in href
            or "/buy/" in href
        ):
            vehicle_links.append(href)

    vehicle_links = list(dict.fromkeys(vehicle_links))

    print("\nPossible vehicle links:", len(vehicle_links))

    for link in vehicle_links[:10]:
        print(link)

print("\nInspection completed.")
