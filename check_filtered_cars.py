
from bs4 import BeautifulSoup

with open("cars_filtered_test.html", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

cards = soup.select("fuse-card")

print("Total fuse-card elements:", len(cards))

print("\nFIRST 10 CARDS")
print("=" * 60)

for i, card in enumerate(cards[:10], start=1):
    print(f"\nCARD {i}")
    print("-" * 40)

    print("Text:", card.get_text(" ", strip=True)[:500])

    links = card.select('a[href*="/vehicledetail/"]')

    print("Vehicle links:", len(links))

    for link in links[:2]:
        print("URL:", link.get("href"))
        print("Link text:", link.get_text(" ", strip=True)[:150])

print("\nNISSAN ALTIMA CHECK")
print("=" * 60)

altima_count = 0

for card in cards:
    text = card.get_text(" ", strip=True)

    if "nissan" in text.lower() and "altima" in text.lower():
        altima_count += 1

print("Cards mentioning Nissan Altima:", altima_count)
