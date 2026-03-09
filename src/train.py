"""
train.py
--------
Trains and evaluates three models, then saves the best one.
Run: python src/train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_processing import prepare_dataset

DATA_PATH = "data/train.csv"
MODEL_DIR = "models"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, predict, and print metrics for one model."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    error = rmse(y_test, preds)
    print(f"  {name:25s} → RMSE: ${error:>10,.0f}   R²: {r2:.4f}")
    return model, preds, error, r2


def plot_feature_importance(model, feature_names, save_path):
    """Save a feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]  # top 15

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color="#2563eb", alpha=0.85)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Feature Importances (XGBoost)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved feature importance chart → {save_path}")


def plot_pred_vs_actual(y_test, preds, save_path):
    """Save predicted vs actual scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, preds, alpha=0.4, color="#2563eb", edgecolors="white", linewidths=0.3, s=40)
    max_val = max(y_test.max(), preds.max())
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Predicted vs Actual Sale Prices", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved prediction chart → {save_path}")


def tune_xgboost(X_train, y_train):
    """Quick GridSearchCV over a small hyperparameter grid."""
    print("\n🔧 Tuning XGBoost hyperparameters...")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    xgb = XGBRegressor(random_state=42, verbosity=0)
    gs = GridSearchCV(xgb, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    return gs.best_estimator_


def main():
    print("=" * 55)
    print("  🏠 House Price Predictor — Training Pipeline")
    print("=" * 55)

    # ── 1. Load & prepare data ─────────────────────────────
    print("\n📂 Loading data...")
    X, y, encoders = prepare_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── 2. Baseline models ────────────────────────────────
    print("\n📈 Training baseline models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        m, preds, error, r2 = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results[name] = {"model": m, "preds": preds, "rmse": error, "r2": r2}

    # ── 3. Tune XGBoost ───────────────────────────────────
    best_xgb = tune_xgboost(X_train, y_train)
    _, preds_xgb, error_xgb, r2_xgb = evaluate_model(
        "XGBoost (tuned)", best_xgb, X_train, X_test, y_train, y_test
    )
    results["XGBoost"] = {"model": best_xgb, "preds": preds_xgb, "rmse": error_xgb, "r2": r2_xgb}

    # ── 4. Cross-validation on best model ─────────────────
    print("\n🔁 5-fold cross-validation (XGBoost)...")
    cv_scores = cross_val_score(best_xgb, X, y, cv=5, scoring="neg_root_mean_squared_error")
    print(f"  CV RMSE: ${-cv_scores.mean():,.0f} ± ${cv_scores.std():,.0f}")

    # ── 5. Save plots ─────────────────────────────────────
    print("\n🎨 Saving plots...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_feature_importance(best_xgb, list(X.columns), f"{MODEL_DIR}/feature_importance.png")
    plot_pred_vs_actual(y_test, preds_xgb, f"{MODEL_DIR}/pred_vs_actual.png")

    # ── 6. Save model & encoders ──────────────────────────
    print("\n💾 Saving model artifacts...")
    joblib.dump(best_xgb, f"{MODEL_DIR}/xgboost_model.pkl")
    joblib.dump(encoders,  f"{MODEL_DIR}/encoders.pkl")
    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_names.pkl")
    print(f"  Saved → {MODEL_DIR}/xgboost_model.pkl")
    print(f"  Saved → {MODEL_DIR}/encoders.pkl")

    print("\n✅ Training complete!\n")


if __name__ == "__main__":
    main()
