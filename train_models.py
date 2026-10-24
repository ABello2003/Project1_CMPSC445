
# ============================================================
# USED VEHICLE VALUE RETENTION - MACHINE LEARNING PROJECT
# ============================================================
# Models:
#   1. Ridge Regression
#   2. Random Forest Regression
#   3. Gradient Boosting Regression
#
# Outputs:
#   - Cleaned modeling dataset
#   - Dataset statistics
#   - Source and feature requirement checks
#   - MAE, RMSE, R2 comparison
#   - Feature importance
#   - Visualizations
#   - Trained best model
# ============================================================

from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

INPUT_FILE = DATA / "used_cars_with_msrp.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {INPUT_FILE}\n"
        "Run python match_msrp.py first."
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("USED VEHICLE VALUE RETENTION - ML PIPELINE")
print("=" * 65)
print(f"Original listings: {len(df)}")

# ============================================================
# 2. VERIFY ORIGINAL DATA REQUIREMENTS
# ============================================================

# These are collected fields, not engineered features.
original_attributes = [
    "source",
    "year",
    "make",
    "model",
    "trim",
    "price",
    "mileage",
    "color",
    "condition",
    "vin",
    "sku",
    "title",
    "description",
    "listing_url"
]

available_original = [
    c for c in original_attributes if c in df.columns
]

print("\nORIGINAL ATTRIBUTE CHECK")
print(f"Available original attributes: {len(available_original)}")
print(", ".join(available_original))

if len(available_original) < 10:
    print("WARNING: Fewer than 10 original attributes detected.")

# Non-null coverage helps assess whether fields contain useful data.
coverage = pd.DataFrame({
    "attribute": available_original,
    "non_null_count": [
        int(df[c].notna().sum()) for c in available_original
    ],
    "non_null_percent": [
        round(df[c].notna().mean() * 100, 2)
        for c in available_original
    ]
})

coverage.to_csv(
    RESULTS / "original_attribute_coverage.csv",
    index=False
)

# ============================================================
# 3. SOURCE COUNTS BEFORE MODEL FILTERING
# ============================================================

if "source" not in df.columns:
    raise ValueError("Dataset is missing the source column.")

source_counts = df["source"].value_counts()

print("\nSOURCE COUNTS - INTEGRATED DATASET")
print(source_counts.to_string())

source_counts.rename_axis("source").reset_index(
    name="listings"
).to_csv(RESULTS / "source_counts_all.csv", index=False)

if (source_counts < 500).any():
    print(
        "\nWARNING: At least one source has fewer than "
        "500 listings in the integrated dataset."
    )

# ============================================================
# 4. CLEAN NUMERIC VALUES
# ============================================================

numeric_cols = [
    "year",
    "price",
    "mileage",
    "original_msrp",
    "value_retention_ratio"
]

for col in numeric_cols:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

    df[col] = pd.to_numeric(df[col], errors="coerce")

# Retain only records with a valid MSRP-based target.
df = df[
    df["price"].gt(0)
    & df["original_msrp"].gt(0)
    & df["year"].notna()
].copy()

# Recalculate the target to guarantee consistency.
df["value_retention_ratio"] = (
    df["price"] / df["original_msrp"]
)

df = df[
    np.isfinite(df["value_retention_ratio"])
    & df["value_retention_ratio"].gt(0)
].copy()

# Keep retention > 1: a listing can exceed its reference
# base MSRP, especially when the vehicle has a higher trim.

# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

before_dedup = len(df)

# Prefer listing URLs when available.
if "listing_url" in df.columns:
    urls = df["listing_url"].fillna("").astype(str).str.strip()
    has_url = urls.ne("")
    with_url = df.loc[has_url].drop_duplicates(
        subset=["source", "listing_url"]
    )
    without_url = df.loc[~has_url]
    df = pd.concat([with_url, without_url])

# Also remove repeated VIN records within each source
# only when VIN is present.
if "vin" in df.columns:
    vins = df["vin"].fillna("").astype(str).str.strip().str.upper()
    has_vin = vins.str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}")
    with_vin = df.loc[has_vin].drop_duplicates(
        subset=["source", "vin"]
    )
    without_vin = df.loc[~has_vin]
    df = pd.concat([with_vin, without_vin])

df = df.reset_index(drop=True)

print(f"\nValid matched listings after cleaning: {len(df)}")
print(f"Duplicate rows removed: {before_dedup - len(df)}")

if len(df) < 1000:
    print(
        "WARNING: Fewer than 1,000 usable MSRP-matched "
        "listings remain after cleaning."
    )

print("\nMATCHED LISTINGS BY SOURCE")
print(df["source"].value_counts().to_string())

df["source"].value_counts().rename_axis(
    "source"
).reset_index(name="listings").to_csv(
    RESULTS / "source_counts_matched.csv",
    index=False
)

# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

# Description-derived features:
# 1. One-owner claim
# 2. Accident-free claim
# 3. Certified pre-owned claim
# 4. AWD / all-wheel-drive mention

if "description" not in df.columns:
    df["description"] = ""

descriptions = df["description"].fillna("").astype(str).str.lower()


patterns = {
    "mentions_one_owner":
        r"\b(?:one[\s-]?owner|1[\s-]?owner|single owner)\b",

    "mentions_accident_free":
        r"\b(?:no accidents?|accident[\s-]?free|"
        r"no accident history|clean carfax)\b",

    "mentions_certified":
        r"\b(?:certified|cpo)\b",

    "mentions_awd":
        r"\b(?:awd|all[\s-]?wheel[\s-]?drive)\b"
}


for feature, pattern in patterns.items():
    df[feature] = descriptions.str.contains(
        pattern, regex=True
    ).astype(int)

# Use listing model year and a consistent 2026 reference year.
# Do not derive age from the scraping date unless available.
REFERENCE_YEAR = 2026

df["vehicle_age"] = (
    REFERENCE_YEAR - df["year"]
).clip(lower=0)

df["mileage_per_year"] = (
    df["mileage"] /
    df["vehicle_age"].clip(lower=1)
)

# ============================================================
# 7. SELECT FEATURES
# ============================================================

# Trim is intentionally excluded, as discussed.
# MSRP is intentionally excluded from X because it is used
# directly in the target definition and could dominate
# the model through a mechanical relationship.
#
# Price, VIN, listing URL, title, and description are also
# excluded as direct predictors.


numeric_features = [
    "vehicle_age",
    "mileage",
    "mileage_per_year",
    "mentions_certified",
    "mentions_awd"
]

categorical_features = [
    "make",
    "model",
    "source"
]

frequency_features = ["color"]


for col in numeric_features:
    if col not in df.columns:
        df[col] = np.nan

for col in categorical_features + frequency_features:
    if col not in df.columns:
        df[col] = "Unknown"

X = df[
    numeric_features
    + categorical_features
    + frequency_features
].copy()

y = df["value_retention_ratio"].copy()

# ============================================================
# 8. CUSTOM FREQUENCY ENCODER
# ============================================================

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replace categorical values with their frequency in
    the training data.

    Learned only from training folds/data, preventing
    test-set information leakage.
    """

    def fit(self, X, y=None):
        values = pd.DataFrame(X).fillna("Unknown")
        self.maps_ = [
            values.iloc[:, i].astype(str).value_counts(
                normalize=True
            ).to_dict()
            for i in range(values.shape[1])
        ]
        return self

    def transform(self, X):
        values = pd.DataFrame(X).fillna("Unknown")
        columns = []

        for i, mapping in enumerate(self.maps_):
            encoded = (
                values.iloc[:, i].astype(str)
                .map(mapping)
                .fillna(0.0)
                .to_numpy()
            )
            columns.append(encoded)

        return np.column_stack(columns)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [
                f"frequency_{i}"
                for i in range(len(self.maps_))
            ]
        return np.asarray([
            f"{col}_frequency" for col in input_features
        ], dtype=object)


# ============================================================
# 9. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline,
     categorical_features),
    ("frequency", FrequencyEncoder(),
     frequency_features)
])

# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

if len(df) < 30:
    raise ValueError("Not enough records to train models.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN / TEST SPLIT")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================
# 11. TRAIN THREE REGRESSION MODELS
# ============================================================

algorithms = {
    "Ridge Regression": Ridge(alpha=10.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

trained_models = {}
predictions = {}
results = []

print("\nMODEL TRAINING RESULTS")
print("-" * 65)

for name, algorithm in algorithms.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )
    r2 = r2_score(y_test, y_pred)

    trained_models[name] = model
    predictions[name] = y_pred

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{name}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

results_df.to_csv(
    RESULTS / "model_comparison.csv",
    index=False
)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nBEST MODEL BY TEST RMSE:", best_name)

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

# Permutation importance measures the change in test
# performance when one original input feature is shuffled.
# It works across all three models and retains meaningful
# original feature names instead of expanded dummy columns.

from sklearn.inspection import permutation_importance

importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=5,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    RESULTS / "feature_importance.csv",
    index=False
)

print("\nTOP FEATURES")
print(importance_df.head(10).to_string(index=False))

# ============================================================
# 13. SAVE DATASET AND MODEL
# ============================================================

df.to_csv(
    DATA / "used_cars_model_ready.csv",
    index=False
)

joblib.dump(
    best_model,
    RESULTS / "best_retention_model.joblib"
)

# ============================================================
# 14. VISUALIZATIONS
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        PLOTS / filename,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()


# Plot 1: Listing price distribution
plt.figure(figsize=(9, 5))
plt.hist(df["price"], bins=35, edgecolor="black")
plt.title("Used Vehicle Listing Price Distribution")
plt.xlabel("Listing Price ($)")
plt.ylabel("Number of Vehicles")
save_plot("01_price_distribution.png")


# Plot 2: Mileage distribution
plt.figure(figsize=(9, 5))
plt.hist(
    df["mileage"].dropna(),
    bins=35,
    edgecolor="black"
)
plt.title("Vehicle Mileage Distribution")
plt.xlabel("Mileage")
plt.ylabel("Number of Vehicles")
save_plot("02_mileage_distribution.png")


# Plot 3: Listings by make and model
counts = (
    df.groupby(["make", "model"])
    .size()
    .sort_values()
)

plt.figure(figsize=(10, 6))
counts.plot(kind="barh")
plt.title("Used Vehicle Listings by Make and Model")
plt.xlabel("Number of Vehicles")
save_plot("03_make_model_frequencies.png")


# Plot 4: Retention ratio distribution
plt.figure(figsize=(9, 5))
plt.hist(
    df["value_retention_ratio"],
    bins=35,
    edgecolor="black"
)
plt.title("Base-MSRP-Relative Value Retention")
plt.xlabel("Listing Price / Base MSRP")
plt.ylabel("Number of Vehicles")
save_plot("04_retention_distribution.png")


# Plot 5: Model RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Regression Model Performance - RMSE")
plt.ylabel("RMSE (Retention Ratio Units)")
plt.xticks(rotation=15)
save_plot("05_model_rmse.png")


# Plot 6: Model R2 comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Regression Model Performance - R2")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
save_plot("06_model_r2.png")


# Plot 7: Feature importance
top = importance_df.head(12).iloc[::-1]

plt.figure(figsize=(10, 7))
plt.barh(top["Feature"], top["Importance"])
plt.title(f"Feature Importance - {best_name}")
plt.xlabel("Increase in RMSE When Feature Is Shuffled")
save_plot("07_feature_importance.png")


# Plot 8: Actual vs predicted
best_pred = predictions[best_name]

plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_pred, alpha=0.5)

minimum = min(y_test.min(), best_pred.min())
maximum = max(y_test.max(), best_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    color="black"
)

plt.title(f"Actual vs Predicted - {best_name}")
plt.xlabel("Actual Retention Ratio")
plt.ylabel("Predicted Retention Ratio")
save_plot("08_actual_vs_predicted.png")


# Plot 9: Vehicle age and retention
age_summary = (
    df.groupby("vehicle_age")["value_retention_ratio"]
    .agg(["mean", "count"])
    .reset_index()
)

# Avoid drawing a trend based on only one or two vehicles.
age_summary = age_summary[
    age_summary["count"] >= 5
]

plt.figure(figsize=(9, 5))
plt.plot(
    age_summary["vehicle_age"],
    age_summary["mean"],
    marker="o"
)
plt.title("Average Retention Ratio by Vehicle Age")
plt.xlabel("Vehicle Age (Years)")
plt.ylabel("Average Retention Ratio")
save_plot("09_depreciation_by_age.png")


# Plot 10: Mileage and retention
plt.figure(figsize=(9, 5))
plt.scatter(
    df["mileage"],
    df["value_retention_ratio"],
    alpha=0.3
)
plt.title("Mileage vs Value Retention")
plt.xlabel("Mileage")
plt.ylabel("Retention Ratio")
save_plot("10_mileage_vs_retention.png")


# ============================================================
# 15. SUMMARY STATISTICS
# ============================================================

df.describe(include="all").transpose().to_csv(
    RESULTS / "dataset_statistics.csv"
)

description_feature_counts = {
    feature: int(df[feature].sum())
    for feature in patterns
}

pd.DataFrame(
    list(description_feature_counts.items()),
    columns=["Feature", "Positive_Listings"]
).to_csv(
    RESULTS / "description_feature_counts.csv",
    index=False
)

# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PROJECT SUMMARY")
print("=" * 65)

print(f"Usable listings: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Original attributes detected: {len(available_original)}")
print(f"Best model: {best_name}")

print("\nDescription features:")
for feature, count in description_feature_counts.items():
    print(f"  {feature}: {count}")

print("\nOutput folders:")
print(f"  {RESULTS}")
print(f"  {PLOTS}")

print("\nSaved files:")
print("  data/used_cars_model_ready.csv")
print("  results/model_comparison.csv")
print("  results/feature_importance.csv")
print("  results/dataset_statistics.csv")
print("  results/original_attribute_coverage.csv")
print("  results/source_counts_all.csv")
print("  results/source_counts_matched.csv")
print("  results/description_feature_counts.csv")
print("  results/best_retention_model.joblib")
print("  results/plots/ (10 charts)")

print("\nPIPELINE COMPLETE")
