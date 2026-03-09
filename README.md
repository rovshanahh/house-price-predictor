# 🏠 House Price Predictor

An end-to-end machine learning project that predicts house prices using the Ames Housing dataset. Covers the full ML pipeline: data cleaning, EDA, model training, and deployment as a live web app.

**[🚀 Live Demo](https://your-app.onrender.com)** · **[📓 EDA Notebook](notebooks/01_eda.ipynb)** · **[📊 Model Results](notebooks/02_modeling.ipynb)**

---

## ✨ Features

- **Exploratory Data Analysis** — missing value analysis, correlation heatmaps, distribution plots
- **Feature Engineering** — encoding, imputation, outlier handling
- **Model Comparison** — Linear Regression vs Random Forest vs XGBoost
- **Hyperparameter Tuning** — GridSearchCV optimization
- **REST API** — FastAPI backend with `/predict` endpoint
- **Web Interface** — clean HTML form for live predictions

---

## 📊 Results

| Model              | RMSE      | R² Score |
|--------------------|-----------|----------|
| Linear Regression  | ~$28,000  | 0.81     |
| Random Forest      | ~$21,000  | 0.88     |
| XGBoost            | ~$19,500  | 0.90     |

---

## 🗂️ Project Structure

```
house-price-predictor/
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_modeling.ipynb     # Model training & evaluation
├── src/
│   ├── data_processing.py    # Cleaning & feature engineering
│   ├── train.py              # Model training & saving
│   └── predict.py            # Inference logic
├── static/
│   └── index.html            # Web UI
├── models/                   # Saved model artifacts (git-ignored)
├── data/                     # Raw data (git-ignored)
├── main.py                   # FastAPI app
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
```

### 2. Get the Data

Download `train.csv` from [Kaggle - House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place it in `data/`.

### 3. Train the Model

```bash
python src/train.py
```

This saves the trained model to `models/xgboost_model.pkl`.

### 4. Run the API

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000` to use the web interface, or `http://localhost:8000/docs` for the interactive API docs.

---

## 🌐 Deployment (Render)

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set **Build Command**: `pip install -r requirements.txt && python src/train.py`
4. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Done — Render gives you a free public URL!

---

## 🛠️ Tech Stack

- **Data**: `pandas`, `numpy`
- **ML**: `scikit-learn`, `xgboost`
- **Visualization**: `matplotlib`, `seaborn`
- **API**: `FastAPI`, `uvicorn`
- **Deployment**: Render (free tier)

---

## 📚 Dataset

[Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — 79 features describing residential homes in Ames, Iowa. Originally compiled by Dean De Cock for use in data science education.
