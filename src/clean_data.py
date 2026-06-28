from __future__ import annotations

import pandas as pd


# ==============================
# 1. TIMESTAMP NORMALIZATION
# ==============================
def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    working_df["timestamp"] = pd.to_datetime(
        working_df["timestamp"], errors="coerce"
    )

    working_df = working_df.dropna(subset=["timestamp"])

    return working_df


# ==============================
# 2. AMOUNT NORMALIZATION
# ==============================
def normalize_amount(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    working_df["amount"] = pd.to_numeric(
        working_df["amount"], errors="coerce"
    )

    return working_df


# ==============================
# 3. STRING CLEANING
# ==============================
def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    string_cols = [
        "currency",
        "transaction_type",
        "country",
        "channel",
        "status",
        "account_status_change",
    ]

    for col in string_cols:
        if col in working_df.columns:
            working_df[col] = (
                working_df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    return working_df


# ==============================
# 4. COUNTRY STANDARDIZATION
# ==============================
def standardize_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    mapping = {
        "turkey": "tr",
        "tr": "tr",
        "united states": "us",
        "usa": "us",
        "germany": "de",
    }

    if "country" in working_df.columns:
        working_df["country"] = working_df["country"].replace(mapping)

    return working_df


# ==============================
# 5. REMOVE DUPLICATES
# ==============================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    before = len(working_df)
    working_df = working_df.drop_duplicates()
    after = len(working_df)

    print(f"[CLEANING] Removed {before - after} duplicate rows")

    return working_df


# ==============================
# 6. DERIVED FEATURES
# ==============================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    working_df["hour"] = working_df["timestamp"].dt.hour
    working_df["is_night"] = working_df["hour"].apply(
        lambda x: 1 if 0 <= x <= 5 else 0
    )

    working_df["amount_abs"] = working_df["amount"].abs()

    return working_df


# ==============================
# 7. OUTLIER DETECTION
# ==============================
def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    if working_df["amount"].notna().sum() == 0:
        working_df["is_outlier"] = False
        return working_df

    threshold = working_df["amount"].quantile(0.99)

    working_df["is_outlier"] = working_df["amount"] > threshold

    return working_df


# ==============================
# 8. VALIDATION LAYER
# ==============================
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    working_df["missing_timestamp"] = working_df["timestamp"].isna()
    working_df["missing_amount"] = working_df["amount"].isna()
    working_df["invalid_amount"] = working_df["amount"] <= 0

    return working_df


# ==============================
# 9. MAIN PIPELINE
# ==============================
def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full data cleaning and preprocessing pipeline.

    Steps:
    - timestamp normalization
    - amount normalization
    - string cleaning
    - country standardization
    - duplicate removal
    - feature engineering
    - outlier detection
    - validation flags
    """

    df = normalize_timestamps(df)
    df = normalize_amount(df)
    df = clean_strings(df)
    df = standardize_country_codes(df)
    df = remove_duplicates(df)
    df = add_derived_features(df)
    df = flag_outliers(df)
    df = validate_dataset(df)

    return df