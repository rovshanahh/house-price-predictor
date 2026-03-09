"""
data_processing.py
------------------
Handles all data cleaning and feature engineering for the Ames Housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Features we'll use for the model (subset of the 79 available)
NUMERIC_FEATURES = [
    "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
    "GarageArea", "LotArea", "YearBuilt", "YearRemodAdd",
    "OverallQual", "OverallCond", "BedroomAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "MasVnrArea", "WoodDeckSF",
    "OpenPorchSF", "BsmtFinSF1", "BsmtUnfSF", "FullBath", "HalfBath"
]

CATEGORICAL_FEATURES = [
    "Neighborhood", "BldgType", "HouseStyle", "RoofStyle",
    "Exterior1st", "Foundation", "Heating", "CentralAir",
    "SaleType", "SaleCondition", "MSZoning", "LotShape"
]

TARGET = "SalePrice"


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing missing value counts and percentages."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct})


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Fill missing numeric values with median
    - Fill missing categorical values with 'Unknown'
    - Remove extreme outliers in GrLivArea
    """
    df = df.copy()

    # Drop obvious outlier rows (per Ames Housing dataset documentation)
    if "GrLivArea" in df.columns:
        df = df[~((df["GrLivArea"] > 4000) & (df.get("SalePrice", pd.Series([1])) < 300000))]

    # Fill numeric NaNs with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs with 'Unknown'
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    print(f"✅ Cleaned data: {len(df)} rows remaining")
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encode categorical features.
    
    Args:
        df: Input DataFrame
        encoders: Dict of pre-fitted LabelEncoders (used during inference)
        fit: If True, fit new encoders. If False, use provided encoders.

    Returns:
        (transformed DataFrame, encoders dict)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    return df, encoders


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the features used for modeling."""
    cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    return df[cols]


def prepare_dataset(filepath: str):
    """
    Full pipeline: load → clean → encode → split features/target.
    Returns X, y, encoders.
    """
    df = load_data(filepath)
    df = clean_data(df)
    df, encoders = encode_categoricals(df, fit=True)
    X = get_features(df)
    y = df[TARGET] if TARGET in df.columns else None
    return X, y, encoders
