
import re
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

CARS_FILE = DATA_DIR / "cars_collected.csv"
CARVANA_FILE = DATA_DIR / "carvana_collected.csv"

OUTPUT_FILE = DATA_DIR / "used_cars_cleaned.csv"
REPORT_FILE = DATA_DIR / "data_quality_report.txt"
REJECTED_FILE = DATA_DIR / "rejected_vehicles.csv"
DUPLICATES_FILE = DATA_DIR / "duplicate_vehicles.csv"

CURRENT_YEAR = date.today().year

TARGET_MODELS = {
    "Nissan": ["Altima", "Sentra", "Rogue"],
    "Toyota": ["Camry", "Corolla", "RAV4"],
    "BMW": ["3 Series", "5 Series", "X3"]
}

COLUMNS = [
    "source", "year", "make", "model", "trim",
    "price", "mileage", "color", "condition",
    "vin", "sku", "title", "description",
    "listing_url"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def clean_numeric(series):
    return pd.to_numeric(
        series.astype("string").str.replace(
            r"[$,]", "", regex=True
        ),
        errors="coerce"
    )


def normalize_make(value):
    value = clean_text(value).lower()

    mapping = {
        "nissan": "Nissan",
        "toyota": "Toyota",
        "bmw": "BMW"
    }

    return mapping.get(value, value.title())


def normalize_model(row):
    make = row["make"]
    model = clean_text(row["model"])

    for target in TARGET_MODELS.get(make, []):
        if model.lower() == target.lower():
            return target

    return model


def normalize_vin(value):
    vin = clean_text(value).upper()

    if re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
        return vin

    return ""


def normalize_url(value):
    url = clean_text(value)

    if not url:
        return ""

    return url.split("?")[0].rstrip("/").lower()


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path, source):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "vin": "string",
            "sku": "string",
            "listing_url": "string"
        }
    )

    print(f"{source}: {len(df)} records loaded")

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df["source"] = source

    return df[COLUMNS].copy()


# ============================================================
# MAIN DATA PREPARATION
# ============================================================

def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("USED CAR DATA PREPARATION")
    print("=" * 60)

    cars = load_dataset(CARS_FILE, "Cars.com")
    carvana = load_dataset(CARVANA_FILE, "Carvana")

    df = pd.concat(
        [cars, carvana],
        ignore_index=True
    )

    original_count = len(df)

    print("\nCombined records:", original_count)

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    text_columns = [
        "source", "make", "model", "trim",
        "color", "condition", "vin", "sku",
        "title", "description", "listing_url"
    ]

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    df["make"] = df["make"].apply(normalize_make)

    df["model"] = df.apply(
        normalize_model, axis=1
    )

    df["vin"] = df["vin"].apply(normalize_vin)

    df["listing_url"] = df["listing_url"].apply(
        normalize_url
    )

    # --------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------------

    df["year"] = clean_numeric(df["year"])
    df["price"] = clean_numeric(df["price"])
    df["mileage"] = clean_numeric(df["mileage"])

    # --------------------------------------------------------
    # VALIDATE VEHICLES
    # --------------------------------------------------------

    valid_model = df.apply(
        lambda row: row["model"] in TARGET_MODELS.get(
            row["make"], []
        ),
        axis=1
    )

    valid_year = df["year"].between(
        2000, CURRENT_YEAR
    )

    valid_price = df["price"].between(
        1000, 250000
    )

    valid_mileage = df["mileage"].between(
        0, 400000
    )

    valid_url = df["listing_url"].str.startswith(
        ("http://", "https://")
    )

    valid = (
        valid_model
        & valid_year
        & valid_price
        & valid_mileage
        & valid_url
    )

    rejected = df[~valid].copy()
    df = df[valid].copy()

    rejected.to_csv(
        REJECTED_FILE, index=False
    )

    print("Invalid records removed:", len(rejected))

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    duplicates = []

    # Duplicate listing URLs
    duplicate_url = df.duplicated(
        subset=["listing_url"],
        keep="first"
    )

    duplicates.append(df[duplicate_url].copy())
    df = df[~duplicate_url].copy()

    # Duplicate VINs across sources.
    # Preserve one record per VIN in the modeling dataset.
    has_vin = df["vin"].ne("")

    duplicate_vin = (
        has_vin
        & df.duplicated(
            subset=["vin"],
            keep="first"
        )
    )

    duplicates.append(df[duplicate_vin].copy())
    df = df[~duplicate_vin].copy()

    duplicate_df = pd.concat(
        duplicates,
        ignore_index=True
    )

    duplicate_df.to_csv(
        DUPLICATES_FILE, index=False
    )

    print("Duplicate records removed:", len(duplicate_df))

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    df["year"] = df["year"].astype(int)
    df["price"] = df["price"].astype(float)
    df["mileage"] = df["mileage"].astype(int)

    df["vehicle_age"] = (
        CURRENT_YEAR - df["year"]
    )

    df["mileage_per_year"] = (
        df["mileage"]
        / df["vehicle_age"].clip(lower=1)
    )

    # These are text-derived indicators, not verified
    # vehicle history reports.
    description = df["description"].str.lower()

    df["mentions_one_owner"] = (
        description.str.contains(
            r"\b(?:one.owner|1.owner|single.owner)\b",
            regex=True,
            na=False
        ).astype(int)
    )

    df["mentions_accident_free"] = (
        description.str.contains(
            r"\b(?:no accidents?|accident.free|"
            r"clean carfax|no accident reported)\b",
            regex=True,
            na=False
        ).astype(int)
    )

    df["mentions_certified"] = (
        description.str.contains(
            r"\b(?:certified pre.owned|cpo|certified)\b",
            regex=True,
            na=False
        ).astype(int)
    )

    df["mentions_awd"] = (
        (
            df["description"] + " " + df["trim"]
        ).str.contains(
            r"\b(?:awd|all.wheel.drive|xdrive|4matic)\b",
            regex=True,
            case=False,
            na=False
        ).astype(int)
    )

    # MSRP is deliberately left missing until verified.
    df["original_msrp"] = np.nan
    df["value_retention_ratio"] = np.nan

    # --------------------------------------------------------
    # SAVE CLEANED DATASET
    # --------------------------------------------------------

    df = df.sort_values(
        ["make", "model", "year", "price"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE, index=False
    )

    # --------------------------------------------------------
    # GENERATE QUALITY REPORT
    # --------------------------------------------------------

    missing_values = df.isna().sum()

    report = []

    report.append("USED CAR DATA QUALITY REPORT")
    report.append("=" * 60)

    report.append(
        f"Original combined records: {original_count}"
    )

    report.append(
        f"Invalid records removed: {len(rejected)}"
    )

    report.append(
        f"Duplicate records removed: {len(duplicate_df)}"
    )

    report.append(
        f"Final cleaned records: {len(df)}"
    )

    report.append("\nRECORDS BY SOURCE")
    report.append(
        df["source"].value_counts().to_string()
    )

    report.append("\nRECORDS BY MAKE AND MODEL")
    report.append(
        df.groupby(["make", "model"])
        .size()
        .to_string()
    )

    report.append("\nMISSING VALUES")
    report.append(
        missing_values.to_string()
    )

    report.append("\nPRICE STATISTICS")
    report.append(
        df["price"].describe().to_string()
    )

    report.append("\nMILEAGE STATISTICS")
    report.append(
        df["mileage"].describe().to_string()
    )

    report.append("\nFEATURE INDICATOR COUNTS")

    for column in [
        "mentions_one_owner",
        "mentions_accident_free",
        "mentions_certified",
        "mentions_awd"
    ]:
        report.append(
            f"{column}: {df[column].sum()}"
        )

    report_text = "\n".join(report)

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print("Original combined:", original_count)
    print("Invalid removed:", len(rejected))
    print("Duplicates removed:", len(duplicate_df))
    print("Final cleaned:", len(df))

    print("\nRecords by source:")
    print(df["source"].value_counts())

    print("\nRecords by make and model:")
    print(
        df.groupby(["make", "model"]).size()
    )

    print("\nFiles generated:")
    print("-", OUTPUT_FILE)
    print("-", REPORT_FILE)
    print("-", REJECTED_FILE)
    print("-", DUPLICATES_FILE)

    print(
        "\nNOTE: Original MSRP and value retention "
        "ratio are not yet calculated."
    )


if __name__ == "__main__":
    main()
