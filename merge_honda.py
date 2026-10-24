
from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd

DATA_DIR = Path("data")

BASE_FILE = DATA_DIR / "used_cars_cleaned.csv"
HONDA_CARS = DATA_DIR / "honda_cars_collected.csv"
HONDA_CARVANA = DATA_DIR / "honda_carvana_collected.csv"

OUTPUT_FILE = DATA_DIR / "used_cars_honda_merged.csv"

ALLOWED_MODELS = {
    "Toyota": {"Camry", "Corolla", "RAV4"},
    "Nissan": {"Altima", "Sentra", "Rogue"},
    "Honda": {"Civic", "Accord", "CR-V"}
}


def read_data(path):
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        dtype={"vin": "string", "sku": "string"}
    )

    print(f"Loaded {len(df)} rows from {path}")
    return df


def clean_data(df):
    if df.empty:
        return df

    df = df.copy()

    for col in ("make", "model", "source", "listing_url"):
        if col not in df.columns:
            df[col] = ""

        df[col] = df[col].fillna("").astype(str).str.strip()

    df["make"] = df["make"].str.title()

    df["model"] = df["model"].replace({
        "Rav4": "RAV4",
        "Cr-V": "CR-V",
        "CRV": "CR-V",
        "Cr V": "CR-V"
    })

    allowed = df.apply(
        lambda row: (
            row["make"] in ALLOWED_MODELS
            and row["model"] in ALLOWED_MODELS[row["make"]]
        ),
        axis=1
    )
    df = df[allowed].copy()

    for col in ("year", "price", "mileage"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["year", "price", "mileage"])

    df = df[
        df["year"].between(1990, 2030)
        & (df["price"] > 0)
        & (df["mileage"] >= 0)
    ].copy()

    df["year"] = df["year"].astype(int)

    # Remove duplicate listing URLs when available.
    has_url = df["listing_url"].ne("")
    with_url = df[has_url].drop_duplicates(
        subset=["listing_url"], keep="last"
    )
    without_url = df[~has_url]

    df = pd.concat(
        [with_url, without_url],
        ignore_index=True
    )

    # Remove duplicate VINs when a valid VIN is available.
    if "vin" in df.columns:
        vin = df["vin"].fillna("").astype(str).str.strip().str.upper()
        valid_vin = vin.str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}")

        df["vin"] = vin.replace("", pd.NA)

        with_vin = df[valid_vin].drop_duplicates(
            subset=["vin"], keep="last"
        )
        without_vin = df[~valid_vin]

        df = pd.concat(
            [with_vin, without_vin],
            ignore_index=True
        )

    return df.reset_index(drop=True)


def main():
    print("=" * 60)
    print("MERGING HONDA WITH TOYOTA AND NISSAN")
    print("=" * 60)

    if not BASE_FILE.exists():
        raise FileNotFoundError(
            f"Required base dataset missing: {BASE_FILE}"
        )

    base = read_data(BASE_FILE)
    honda_cars = read_data(HONDA_CARS)
    honda_carvana = read_data(HONDA_CARVANA)

    if honda_cars.empty and honda_carvana.empty:
        print("\nNo Honda listings collected yet.")
        print("Run the Honda scraping scripts first.")
        return

    # Keep Honda records only from the new collectors.
    if not honda_cars.empty:
        honda_cars = honda_cars[
            honda_cars["make"].astype(str).str.lower() == "honda"
        ].copy()
        honda_cars["source"] = "Cars.com"

    if not honda_carvana.empty:
        honda_carvana = honda_carvana[
            honda_carvana["make"].astype(str).str.lower() == "honda"
        ].copy()
        honda_carvana["source"] = "Carvana"

    combined = pd.concat(
        [base, honda_cars, honda_carvana],
        ignore_index=True
    )

    before = len(combined)
    cleaned = clean_data(combined)

    print(f"\nRows before cleaning: {before}")
    print(f"Rows after cleaning: {len(cleaned)}")
    print(f"Rows removed: {before - len(cleaned)}")

    if OUTPUT_FILE.exists():
        backup = DATA_DIR / (
            "used_cars_honda_merged_backup_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )
        shutil.copy2(OUTPUT_FILE, backup)
        print(f"Previous merged output backed up: {backup}")

    cleaned.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    print("\nTotal vehicles:", len(cleaned))

    print("\nListings by website:")
    source_counts = cleaned.groupby("source").size()
    print(source_counts.to_string())

    print("\nListings by make and model:")
    print(
        cleaned.groupby(["make", "model"])
        .size()
        .to_string()
    )

    print("\nASSIGNMENT REQUIREMENT CHECK")

    for source in ("Cars.com", "Carvana"):
        count = int(source_counts.get(source, 0))
        status = "PASS" if count >= 500 else "NOT YET MET"
        print(f"{source}: {count}/500 - {status}")

    print(f"\nUnique makes: {cleaned['make'].nunique()}")

    for make in ("Toyota", "Nissan", "Honda"):
        models = sorted(
            cleaned.loc[
                cleaned["make"] == make, "model"
            ].unique()
        )
        print(f"{make}: {', '.join(models) or 'NONE'}")

    print(f"\nMerged dataset saved to: {OUTPUT_FILE}")
    print(
        "\nOriginal used_cars_cleaned.csv "
        "was not overwritten."
    )


if __name__ == "__main__":
    main()
