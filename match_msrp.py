
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA = Path(__file__).resolve().parent / "data"

VEHICLES = DATA / "used_cars_honda_merged.csv"
REFERENCES = DATA / "msrp_reference_verified.csv"

OUTPUT = DATA / "used_cars_with_msrp.csv"
UNMATCHED = DATA / "unmatched_msrp.csv"
PRIORITIES = DATA / "msrp_remaining_priorities.csv"
BASE_REFERENCE = DATA / "msrp_base_reference.csv"
SUMMARY = DATA / "msrp_match_summary.txt"

KEYS = ["year", "make", "model"]

# The standard/base trim for each model.
# CR-V starts at EX for 2023-2024 (LX was discontinued
# for those model years).
BASE_TRIMS = {
    ("Honda", "Civic"): "LX",
    ("Honda", "Accord"): "LX",
    ("Honda", "CR-V"): "LX",
    ("Nissan", "Altima"): "2.5 S",
    ("Nissan", "Sentra"): "S",
    ("Nissan", "Rogue"): "S",
    ("Toyota", "Camry"): "LE",
    ("Toyota", "Corolla"): "LE",
    ("Toyota", "RAV4"): "LE"
}

# Prefer the standard two-wheel-drive base configuration.
# "standard" is used for models with one standard drivetrain.
PREFERRED_DRIVETRAINS = {
    "FWD", "2WD", "STANDARD"
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("YEAR / MAKE / MODEL MSRP MATCHING")
print("=" * 65)

if not VEHICLES.exists():
    raise FileNotFoundError(f"Missing: {VEHICLES}")

if not REFERENCES.exists():
    raise FileNotFoundError(f"Missing: {REFERENCES}")

vehicles = pd.read_csv(VEHICLES)
references = pd.read_csv(REFERENCES)

print(f"\nVehicles loaded: {len(vehicles)}")
print(f"Original MSRP reference rows: {len(references)}")

required_vehicle_cols = {
    "year", "make", "model", "price"
}
required_reference_cols = {
    "year", "make", "model", "trim",
    "drivetrain", "original_msrp"
}

if not required_vehicle_cols.issubset(vehicles.columns):
    raise ValueError("Vehicle dataset missing required columns.")

if not required_reference_cols.issubset(references.columns):
    raise ValueError("MSRP database missing required columns.")


# ============================================================
# NORMALIZE DATA
# ============================================================

for df in [vehicles, references]:
    df["year"] = pd.to_numeric(
        df["year"], errors="coerce"
    )

    df["make"] = (
        df["make"].fillna("")
        .astype(str).str.strip().str.title()
    )

    df["model"] = (
        df["model"].fillna("")
        .astype(str).str.strip()
    )

    df["model"] = df["model"].replace({
        "Rav4": "RAV4",
        "Cr-V": "CR-V",
        "CRV": "CR-V"
    })

references["trim"] = (
    references["trim"]
    .fillna("").astype(str).str.strip()
)

references["drivetrain"] = (
    references["drivetrain"]
    .fillna("").astype(str).str.strip().str.upper()
)

references["original_msrp"] = pd.to_numeric(
    references["original_msrp"],
    errors="coerce"
)

vehicles["price"] = pd.to_numeric(
    vehicles["price"], errors="coerce"
)


# ============================================================
# IDENTIFY BASE MSRP CONFIGURATIONS
# ============================================================


def get_base_trim(year, make, model):
    # 2023 CR-V started at EX.
    # 2024 CR-V brought back the LX base trim.
    if make == "Honda" and model == "CR-V" and year == 2023:
        return "EX"

    return BASE_TRIMS.get((make, model))



base_records = []
conflicts = []

for (year, make, model), group in references.groupby(KEYS):

    if pd.isna(year):
        continue

    year = int(year)

    expected_trim = get_base_trim(year, make, model)

    if expected_trim is None:
        continue

    candidates = group[
        group["trim"].str.casefold()
        == expected_trim.casefold()
    ].copy()

    # Do not substitute an upper trim when the base trim
    # is missing from the reference database.
    if candidates.empty:
        continue

    preferred = candidates[
        candidates["drivetrain"].isin(
            PREFERRED_DRIVETRAINS
        )
    ].copy()

    # If only AWD is available, we cannot claim it is the
    # standard FWD base MSRP.
    if preferred.empty:
        continue

    prices = preferred["original_msrp"].dropna()
    prices = prices[prices > 0].unique()

    # Do not silently resolve conflicting base prices.
    if len(prices) != 1:
        conflicts.append({
            "year": year,
            "make": make,
            "model": model,
            "expected_base_trim": expected_trim,
            "candidate_prices": str(sorted(prices.tolist()))
        })
        continue

    chosen = preferred[
        preferred["original_msrp"] == prices[0]
    ].iloc[0]

    base_records.append({
        "year": year,
        "make": make,
        "model": model,
        "base_trim": expected_trim,
        "base_drivetrain": chosen["drivetrain"],
        "original_msrp": float(prices[0]),
        "msrp_source_url": chosen.get(
            "msrp_source_url", ""
        ),
        "msrp_notes": chosen.get(
            "msrp_notes", ""
        ),
        "verification_scope": chosen.get(
            "verification_scope", ""
        )
    })


base_df = pd.DataFrame(
    base_records,
    columns=KEYS + [
        "base_trim",
        "base_drivetrain",
        "original_msrp",
        "msrp_source_url",
        "msrp_notes",
        "verification_scope"
    ]
)

base_df.to_csv(BASE_REFERENCE, index=False)

print(f"Base MSRP configurations found: {len(base_df)}")
print(f"Conflicting configurations: {len(conflicts)}")


# ============================================================
# MATCH BY YEAR + MAKE + MODEL
# ============================================================

# Remove MSRP results from previous runs.
old_columns = [
    "original_msrp",
    "value_retention_ratio",
    "msrp_match_status",
    "reference_options",
    "base_trim",
    "base_drivetrain",
    "msrp_source_url",
    "msrp_notes",
    "verification_scope"
]

vehicles = vehicles.drop(
    columns=old_columns,
    errors="ignore"
)

merged = vehicles.merge(
    base_df,
    on=KEYS,
    how="left",
    validate="many_to_one"
)

merged["msrp_match_status"] = "NO_BASE_MSRP"

valid = (
    merged["original_msrp"].notna()
    & (merged["original_msrp"] > 0)
    & merged["price"].notna()
    & (merged["price"] > 0)
)

merged.loc[
    valid, "msrp_match_status"
] = "MATCHED_BASE_MSRP"

merged["value_retention_ratio"] = (
    merged["price"] / merged["original_msrp"]
)

# Do not calculate retention for invalid prices.
merged.loc[
    ~valid, "value_retention_ratio"
] = float("nan")

# Identify which MSRP values still need independent auditing.
merged["msrp_verification_status"] = "NOT_MATCHED"

merged.loc[
    valid, "msrp_verification_status"
] = merged.loc[
    valid, "verification_scope"
].fillna("NOT_INDEPENDENTLY_AUDITED")

# ============================================================
# SAVE RESULTS
# ============================================================

merged.to_csv(OUTPUT, index=False)

unmatched = merged[
    merged["msrp_match_status"] != "MATCHED_BASE_MSRP"
].copy()

unmatched.to_csv(UNMATCHED, index=False)

priorities = (
    unmatched.groupby(KEYS, dropna=False)
    .size()
    .reset_index(name="vehicles")
    .sort_values("vehicles", ascending=False)
)

priorities.to_csv(PRIORITIES, index=False)

if conflicts:
    pd.DataFrame(conflicts).to_csv(
        DATA / "msrp_base_conflicts.csv",
        index=False
    )

# ============================================================
# SUMMARY
# ============================================================

total = len(merged)
matched = int(valid.sum())
missing = total - matched

rate = (matched / total * 100) if total else 0

lines = [
    "=" * 65,
    "MSRP MATCHING SUMMARY",
    "=" * 65,
    f"Total vehicles: {total}",
    f"MSRP references loaded: {len(references)}",
    f"Base configurations available: {len(base_df)}",
    f"Matched vehicles: {matched}",
    f"Unmatched vehicles: {missing}",
    f"Match rate: {rate:.2f}%",
    "",
    "Matching method: year + make + model",
    "MSRP basis: standard/base trim, standard drivetrain",
    "Retention ratio: listing price / base MSRP",
    "",
    "NOTE: This is a base-MSRP-relative retention ratio.",
    "It is not an exact trim-specific depreciation ratio.",
    "Existing MSRP references may still require source auditing.",
    "",
    "Vehicles by make:",
    merged.groupby("make").size().to_string(),
    "",
    "Matched vehicles by make:",
    merged[valid].groupby("make").size().to_string(),
    "",
    "Remaining missing configurations:",
    str(len(priorities))
]

summary_text = "\n".join(lines)

SUMMARY.write_text(
    summary_text,
    encoding="utf-8"
)

print("\n" + summary_text)

print("\nFILES SAVED:")
print("1. used_cars_with_msrp.csv")
print("2. unmatched_msrp.csv")
print("3. msrp_remaining_priorities.csv")
print("4. msrp_base_reference.csv")
print("5. msrp_match_summary.txt")

print("\nDone.")
