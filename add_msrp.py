
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

main_file = DATA / "msrp_reference_verified.csv"
additions_file = DATA / "msrp_verified_additions_batch2.csv"

if not main_file.exists():
    raise FileNotFoundError(f"Missing: {main_file}")

if not additions_file.exists():
    raise FileNotFoundError(
        f"Missing: {additions_file}\n"
        "Download the additions CSV and place it inside your data folder."
    )

main = pd.read_csv(main_file)
additions = pd.read_csv(additions_file)

required = [
    "year", "make", "model", "trim",
    "drivetrain", "original_msrp"
]

for col in required:
    if col not in main.columns or col not in additions.columns:
        raise ValueError(f"Missing column: {col}")

# Normalize matching keys
for df in (main, additions):
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in ["make", "model", "trim", "drivetrain"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

# Preserve existing records; append only new configurations
keys = ["year", "make", "model", "trim", "drivetrain"]

existing_keys = set(
    map(tuple, main[keys].itertuples(index=False, name=None))
)

new_rows = additions[
    ~additions[keys].apply(tuple, axis=1).isin(existing_keys)
]

combined = pd.concat([main, new_rows], ignore_index=True)

# Save backup before overwriting
backup = DATA / "msrp_reference_before_batch1.csv"
main.to_csv(backup, index=False)

combined.to_csv(main_file, index=False)

print(f"Original references: {len(main)}")
print(f"New references added: {len(new_rows)}")
print(f"Total references: {len(combined)}")
print(f"Saved: {main_file}")
print(f"Backup: {backup}")