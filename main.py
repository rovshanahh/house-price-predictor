"""
main.py
-------
FastAPI application. Run with: uvicorn main:app --reload
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from predict import predict_price

app = FastAPI(
    title="House Price Predictor API",
    description="Predicts Ames, Iowa house prices using XGBoost.",
    version="1.0.0",
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


# ── Request / Response Schemas ────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    # Key numeric features
    GrLivArea:      float = 1500   # Above-ground living area (sq ft)
    TotalBsmtSF:    float = 800    # Basement area (sq ft)
    GarageArea:     float = 400    # Garage size (sq ft)
    LotArea:        float = 8000   # Lot size (sq ft)
    YearBuilt:      int   = 1990
    YearRemodAdd:   int   = 2000
    OverallQual:    int   = 5      # 1–10 quality scale
    OverallCond:    int   = 5      # 1–10 condition scale
    BedroomAbvGr:   int   = 3
    FullBath:       int   = 2
    HalfBath:       int   = 0
    TotRmsAbvGrd:   int   = 6
    Fireplaces:     int   = 1
    GarageCars:     int   = 2

    # Key categorical features
    Neighborhood:   str = "CollgCr"
    BldgType:       str = "1Fam"
    HouseStyle:     str = "2Story"
    CentralAir:     str = "Y"
    MSZoning:       str = "RL"
    Foundation:     str = "PConc"
    SaleType:       str = "WD"
    SaleCondition:  str = "Normal"

    class Config:
        json_schema_extra = {
            "example": {
                "GrLivArea": 1710,
                "TotalBsmtSF": 856,
                "GarageArea": 548,
                "LotArea": 8450,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "OverallQual": 7,
                "OverallCond": 5,
                "BedroomAbvGr": 3,
                "FullBath": 2,
                "HalfBath": 1,
                "TotRmsAbvGrd": 8,
                "Fireplaces": 0,
                "GarageCars": 2,
                "Neighborhood": "CollgCr",
                "BldgType": "1Fam",
                "HouseStyle": "2Story",
                "CentralAir": "Y",
                "MSZoning": "RL",
                "Foundation": "PConc",
                "SaleType": "WD",
                "SaleCondition": "Normal"
            }
        }


class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """
    Predict the sale price of a house given its features.
    Returns the predicted price in USD.
    """
    try:
        price = predict_price(features.model_dump())
        return PredictionResponse(
            predicted_price=price,
            formatted_price=f"${price:,.0f}"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python src/train.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Check if the API is running."""
    return {"status": "ok"}
