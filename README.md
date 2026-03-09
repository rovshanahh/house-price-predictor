# House Price Predictor

End-to-end machine learning project predicting house sale prices using the Ames Housing dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![FastAPI](https://img.shields.io/badge/API-FastAPI-green)

---

## Overview

Covers the full ML pipeline — data cleaning, exploratory analysis, model training, and a live web interface backed by a REST API.

**Model performance**

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | ~$28,000 | 0.81 |
| Random Forest | ~$21,000 | 0.88 |
| XGBoost (tuned) | ~$19,500 | 0.90 |

---

## Project Structure

```
house-price-predictor/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   └── predict.py
├── static/
│   └── index.html
├── main.py
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/rovshanahh/house-price-predictor
cd house-price-predictor
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place it in `data/`.

```bash
python src/train.py
uvicorn main:app --reload
```

Visit `http://localhost:8000`

---

## Stack

Python · pandas · scikit-learn · XGBoost · FastAPI · Uvicorn
