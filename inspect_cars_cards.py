
from bs4 import BeautifulSoup

with open("cars_com_test.html", encoding="utf-8") as file:
    soup = BeautifulSoup(file.read(), "html.parser")

links = soup.find_all(
    "a",
    href=lambda href: href and "/vehicledetail/" in href
)

print("Vehicle links found:", len(links))

for i, link in enumerate(links[:5], start=1):
    print(f"\n{'=' * 60}")
    print(f"VEHICLE {i}")
    print("=" * 60)

    print("URL:", link.get("href"))

    # Inspect the link's surrounding HTML containers.
    parent = link

    for level in range(1, 5):
        parent = parent.parent

        if parent is None:
            break

        print(f"\nPARENT LEVEL {level}")
        print("Tag:", parent.name)
        print("Classes:", parent.get("class"))
        print("Text:", parent.get_text(" ", strip=True)[:600])
