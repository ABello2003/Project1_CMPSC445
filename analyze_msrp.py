
import pandas as pd
from pathlib import Path

# File locations
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "used_cars_cleaned.csv"
MSRP_FILE = DATA_DIR / "msrp_needed.csv"
MISSING_FILE = DATA_DIR / "missing_trim.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("MSRP REQUIREMENTS ANALYSIS")
print("=" * 60)

print(f"\nTotal vehicles: {len(df)}")

# Standardize trim names
df["trim"] = (
    df["trim"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Identify missing or unknown trims
missing_mask = (
    df["trim"].eq("")
    | df["trim"].str.lower().isin(
        ["unknown", "none", "nan", "n/a"]
    )
)

missing_trim = df[missing_mask].copy()
missing_trim.to_csv(MISSING_FILE, index=False)

# Use only known trims for MSRP lookup
known_trim = df[~missing_mask].copy()

# Count unique vehicle configurations
msrp_needed = (
    known_trim.groupby(
        ["year", "make", "model", "trim"],
        dropna=False
    )
    .agg(
        vehicle_count=("price", "size"),
        average_used_price=("price", "mean"),
        min_used_price=("price", "min"),
        max_used_price=("price", "max")
    )
    .reset_index()
)

# Most common configurations first
msrp_needed = msrp_needed.sort_values(
    "vehicle_count",
    ascending=False
)

# Add fields to fill during MSRP research
msrp_needed["original_msrp"] = pd.NA
msrp_needed["msrp_source_url"] = ""
msrp_needed["msrp_notes"] = ""

# Save
msrp_needed.to_csv(MSRP_FILE, index=False)

# Summary
print(f"Vehicles with known trims: {len(known_trim)}")
print(f"Vehicles missing trims: {len(missing_trim)}")
print(f"Unique MSRP combinations: {len(msrp_needed)}")

print("\nTOP 20 MOST COMMON CONFIGURATIONS")
print("-" * 60)

print(
    msrp_needed[
        ["year", "make", "model", "trim", "vehicle_count"]
    ].head(20).to_string(index=False)
)

print("\nCONFIGURATIONS BY MAKE")
print("-" * 60)

print(
    msrp_needed.groupby("make")
    .size()
    .to_string()
)

print("\nFILES CREATED:")
print(MSRP_FILE)
print(MISSING_FILE)

print("\nAnalysis complete.")
