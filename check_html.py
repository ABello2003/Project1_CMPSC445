
from bs4 import BeautifulSoup
from pathlib import Path

files = ["cars_com_test.html", "carvana_test.html"]

for filename in files:
    print(f"\n{'=' * 50}")
    print(f"Checking {filename}")
    print("=" * 50)

    if not Path(filename).exists():
        print("File not found!")
        continue

    with open(filename, "r", encoding="utf-8") as file:
        html = file.read()

    soup = BeautifulSoup(html, "html.parser")

    print("Page title:",
          soup.title.get_text(strip=True) if soup.title else "None")

    text = soup.get_text(" ", strip=True)

    print("\nFirst 500 characters:")
    print(text[:500])

    print("\nVehicle data indicators:")

    for keyword in [
        "Nissan", "Toyota", "BMW",
        "mileage", "price", "vehicle",
        "inventory", "captcha", "access denied"
    ]:
        print(f"{keyword}: {keyword.lower() in html.lower()}")

    print("\nStructured data scripts:",
          len(soup.select('script[type="application/ld+json"]')))

    print("HTML size:", len(html))
