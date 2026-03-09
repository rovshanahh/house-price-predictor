"""
predict.py
----------
Loads the saved model and runs inference on new input data.
Used by the FastAPI app.
"""

import joblib
import pandas as pd
import numpy as np
import os

from data_processing import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    clean_data, encode_categoricals, get_features
)

MODEL_DIR = "models"

_model = None
_encoders = None
_feature_names = None


def _load_artifacts():
    """Lazy-load model artifacts once."""
    global _model, _encoders, _feature_names
    if _model is None:
        _model        = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
        _encoders     = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        _feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


def predict_price(input_dict: dict) -> float:
    """
    Predict the sale price for a single house.

    Args:
        input_dict: Dict with feature names as keys (strings/numbers).

    Returns:
        Predicted price as a float.
    """
    _load_artifacts()

    df = pd.DataFrame([input_dict])

    # Fill any missing features with sensible defaults
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

    # Apply same cleaning + encoding used at training time
    df = clean_data(df)
    df, _ = encode_categoricals(df, encoders=_encoders, fit=False)
    X = get_features(df)

    # Ensure column order matches training
    X = X.reindex(columns=_feature_names, fill_value=0)

    prediction = _model.predict(X)[0]
    return round(float(prediction), 2)
