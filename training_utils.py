"""
Shared utilities for House Price Prediction Model Training & Evaluation.
Provides dataset loading, consistent train/test splitting, preprocessors,
evaluation metric calculation, feature importance aggregation, and metrics persistence.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset location resolver
def get_dataset_path() -> Path:
    candidates = [
        BASE_DIR / "data" / "house_price_dataset_india.csv",
        BASE_DIR.parent / "src" / "assets" / "house_price_dataset_india.csv",
        BASE_DIR.parent / "ml_project" / "src" / "assets" / "house_price_dataset_india.csv",
        Path("C:/Users/shour/Desktop/backend/data/house_price_dataset_india.csv"),
        Path("C:/Users/shour/Desktop/ml_project/src/assets/house_price_dataset_india.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find house_price_dataset_india.csv in candidate paths.")

FEATURE_COLS = [
    'City', 'Locality_Tier', 'BHK', 'Bathrooms', 'Super_Area_sqft', 'Carpet_Area_sqft',
    'Floor_No', 'Total_Floors', 'Property_Age_years', 'Parking', 'Furnishing',
    'Lift', 'Gated_Society', 'Distance_to_Metro_km', 'Distance_to_CityCenter_km',
    'Nearby_School_km', 'Nearby_Hospital_km', 'Crime_Rate_Index'
]
TARGET_COL = 'Market_Price_INR'
CATEGORICAL_COLS = ['City', 'Locality_Tier', 'Furnishing']
NUMERICAL_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

PRICE_TIERS = [
    {"name": "Budget (< ₹60L)", "min": 0, "max": 6_000_000},
    {"name": "Mid-Range (₹60L–₹1Cr)", "min": 6_000_000, "max": 10_000_000},
    {"name": "Premium (₹1Cr–₹1.75Cr)", "min": 10_000_000, "max": 17_500_000},
    {"name": "Luxury (> ₹1.75Cr)", "min": 17_500_000, "max": float("inf")},
]
TIER_LABELS = [t["name"] for t in PRICE_TIERS]

def price_to_tier(prices: np.ndarray) -> np.ndarray:
    labels = []
    for p in prices:
        if p < 6_000_000:
            labels.append("Budget (< ₹60L)")
        elif p < 10_000_000:
            labels.append("Mid-Range (₹60L–₹1Cr)")
        elif p < 17_500_000:
            labels.append("Premium (₹1Cr–₹1.75Cr)")
        else:
            labels.append("Luxury (> ₹1.75Cr)")
    return np.array(labels)

def compute_adjusted_r2(r2: float, n: int, p: int) -> float:
    if n - p - 1 <= 0:
        return r2
    return 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))

def get_preprocessor_scaled() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_COLS)
        ]
    )

def get_preprocessor_tree() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_COLS)
        ],
        remainder='passthrough'
    )

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    csv_path = get_dataset_path()
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_test_tiers = price_to_tier(y_test.values)
    return df, X_train, X_test, y_train, y_test, y_test_tiers

def map_to_base_feature(name: str) -> str:
    for cat in CATEGORICAL_COLS:
        if name == cat or name.startswith(cat + '_'):
            return cat
    return name

def evaluate_model_pipeline(
    model_key: str,
    model_name: str,
    short_name: str,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_test_tiers: np.ndarray,
    train_time: float,
    best_params: Dict[str, Any],
    algorithm_type: str,
    tuning_summary: str
) -> Tuple[Dict[str, Any], np.ndarray]:
    n = len(X_test)
    p = len(FEATURE_COLS)

    t0 = time.perf_counter()
    preds = pipeline.predict(X_test)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / n

    r2 = float(r2_score(y_test, preds))
    adj_r2 = float(compute_adjusted_r2(r2, n, p))
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(root_mean_squared_error(y_test, preds))
    mape = float(mean_absolute_percentage_error(y_test, preds) * 100.0)
    med_ae = float(median_absolute_error(y_test, preds))
    m_err = float(max_error(y_test, preds))
    exp_var = float(explained_variance_score(y_test, preds))

    pred_tiers = price_to_tier(preds)
    cm = confusion_matrix(y_test_tiers, pred_tiers, labels=TIER_LABELS)
    cm_accuracy = float(accuracy_score(y_test_tiers, pred_tiers))

    tier_report = classification_report(
        y_test_tiers, pred_tiers, labels=TIER_LABELS, output_dict=True, zero_division=0
    )
    tier_breakdown = {}
    for t in TIER_LABELS:
        t_data = tier_report.get(t, {})
        tier_breakdown[t] = {
            "precision": float(round(t_data.get("precision", 0.0), 4)),
            "recall": float(round(t_data.get("recall", 0.0), 4)),
            "f1_score": float(round(t_data.get("f1-score", 0.0), 4)),
            "support": int(t_data.get("support", 0))
        }

    matrix_structured = [
        [
            {"actual": TIER_LABELS[i], "predicted": TIER_LABELS[j], "count": int(cm[i][j])}
            for j in range(len(TIER_LABELS))
        ]
        for i in range(len(TIER_LABELS))
    ]

    # City-by-city evaluation
    city_errors = {}
    for city_val in sorted(X_test['City'].unique()):
        city_mask = (X_test['City'] == city_val)
        y_c = y_test[city_mask]
        preds_c = preds[city_mask.values]
        city_errors[city_val] = {
            "count": int(np.sum(city_mask)),
            "mae": float(round(mean_absolute_error(y_c, preds_c), 1)),
            "rmse": float(round(root_mean_squared_error(y_c, preds_c), 1)),
            "r2": float(round(r2_score(y_c, preds_c), 4))
        }

    # Residuals & distribution
    residuals = y_test.values - preds
    hist_counts, hist_edges = np.histogram(residuals / 1e5, bins=30)
    hist_bins = [
        {
            "bin_start": float(round(hist_edges[i], 1)),
            "bin_end": float(round(hist_edges[i+1], 1)),
            "count": int(hist_counts[i])
        }
        for i in range(len(hist_counts))
    ]
    abs_res = np.abs(residuals)
    res_percentiles = {
        "p50": float(round(np.percentile(abs_res, 50), 2)),
        "p75": float(round(np.percentile(abs_res, 75), 2)),
        "p90": float(round(np.percentile(abs_res, 90), 2)),
        "p95": float(round(np.percentile(abs_res, 95), 2)),
    }

    results = {
        "name": model_name,
        "short_name": short_name,
        "algorithm_type": algorithm_type,
        "train_time_sec": float(round(train_time, 2)),
        "latency_ms_per_prediction": float(round(latency_ms, 4)),
        "best_hyperparameters": best_params,
        "tuning_summary": tuning_summary,
        "metrics": {
            "r2_score": float(round(r2, 4)),
            "adjusted_r2": float(round(adj_r2, 4)),
            "mae": float(round(mae, 1)),
            "rmse": float(round(rmse, 1)),
            "mape_percent": float(round(mape, 1)),
            "median_absolute_error": float(round(med_ae, 1)),
            "max_error": float(round(m_err, 1)),
            "explained_variance": float(round(exp_var, 4))
        },
        "confusion_matrix": {
            "tier_accuracy": float(round(cm_accuracy * 100.0, 2)),
            "labels": TIER_LABELS,
            "raw_matrix": cm.tolist(),
            "matrix": matrix_structured,
            "tier_breakdown": tier_breakdown
        },
        "residuals": {
            "histogram": hist_bins,
            "percentiles": res_percentiles
        },
        "city_errors": city_errors
    }
    return results, preds

def generate_feature_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    ranges = {}
    for col in NUMERICAL_COLS:
        s = df[col]
        ranges[col] = {
            "min": float(round(s.min(), 1)),
            "max": float(round(s.max(), 1)),
            "mean": float(round(s.mean(), 1)),
            "median": float(round(s.median(), 1)),
            "std": float(round(s.std(), 1))
        }
    return {
        "features": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "numerical_cols": NUMERICAL_COLS,
        "cities": sorted(df['City'].unique().tolist()),
        "locality_tiers": ['Budget', 'Mid', 'Premium'],
        "furnishing_options": ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'],
        "ranges": ranges,
        "defaults": {
            "City": "Bangalore",
            "Locality_Tier": "Mid",
            "BHK": 2,
            "Bathrooms": 2,
            "Super_Area_sqft": 1050,
            "Carpet_Area_sqft": 820,
            "Floor_No": 5,
            "Total_Floors": 14,
            "Property_Age_years": 5,
            "Parking": 1,
            "Furnishing": "Semi-Furnished",
            "Lift": 1,
            "Gated_Society": 1,
            "Distance_to_Metro_km": 1.8,
            "Distance_to_CityCenter_km": 12.0,
            "Nearby_School_km": 2.5,
            "Nearby_Hospital_km": 2.8,
            "Crime_Rate_Index": 35.0
        }
    }

def update_metrics_store(
    model_key: str,
    eval_dict: Dict[str, Any],
    feature_ranking: List[Dict[str, Any]],
    model_pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
):
    """
    Saves model_pipeline to MODELS_DIR / f"{model_key}.joblib"
    and updates models/metrics.json with this model's data.
    """
    joblib.dump(model_pipeline, MODELS_DIR / f"{model_key}.joblib")
    print(f"Saved model pipeline to: {MODELS_DIR / f'{model_key}.joblib'}")

    metrics_file = MODELS_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    else:
        df, _, _, _, _, _ = load_data()
        meta = generate_feature_metadata(df)
        with open(MODELS_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        metrics_data = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_summary": {
                "total_records": len(df),
                "features_count": len(FEATURE_COLS),
                "target_column": TARGET_COL,
                "holdout_test_samples": len(X_test),
                "train_samples": len(df) - len(X_test),
                "cities_covered": sorted(df['City'].unique().tolist()),
                "price_stats": {
                    "min_lakhs": round(df[TARGET_COL].min() / 1e5, 2),
                    "max_crores": round(df[TARGET_COL].max() / 1e7, 2),
                    "mean_lakhs": round(df[TARGET_COL].mean() / 1e5, 2),
                    "median_lakhs": round(df[TARGET_COL].median() / 1e5, 2),
                }
            },
            "models": {},
            "feature_importances": {},
            "scatter_samples": [],
            "tier_labels": TIER_LABELS,
            "best_overall_model": model_key
        }

    metrics_data["models"][model_key] = eval_dict
    metrics_data["feature_importances"][model_key] = feature_ranking[:12]
    metrics_data["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Update best overall model by r2_score (or fallback to r2)
    best_key = max(
        metrics_data["models"].keys(),
        key=lambda k: metrics_data["models"][k]["metrics"].get("r2_score", metrics_data["models"][k]["metrics"].get("r2", 0.0))
    )
    metrics_data["best_overall_model"] = best_key

    # Re-generate scatter samples for all loaded models
    # Re-generate scatter samples for all loaded models sorted by actual price
    sorted_order = np.argsort(y_test.values)
    sample_indices = sorted_order[np.linspace(0, len(y_test) - 1, 120, dtype=int)]
    scatter_samples = []
    
    # Load all available models
    available_models = {}
    for k in metrics_data["models"].keys():
        m_path = MODELS_DIR / f"{k}.joblib"
        if m_path.exists():
            try:
                available_models[k] = joblib.load(m_path)
            except Exception as e:
                print(f"Warning: could not load {m_path}: {e}")

    sample_X = X_test.iloc[sample_indices]
    sample_y = y_test.iloc[sample_indices].tolist()

    preds_by_model = {}
    for k, m in available_models.items():
        preds_by_model[k] = m.predict(sample_X)

    for idx, actual_val in enumerate(sample_y):
        actual_lakhs = round(float(actual_val) / 1e5, 2)
        sample_item = {
            "id": int(idx + 1),
            "sample_id": int(idx + 1),
            "actual_price": float(actual_val),
            "actual_price_lakhs": actual_lakhs,
            "tier": str(price_to_tier([actual_val])[0]),
        }
        for k in available_models.keys():
            raw_p = float(preds_by_model[k][idx])
            clamped_p = max(500_000.0, raw_p)
            sample_item[k] = round(clamped_p, 2)
            sample_item[f"{k}_lakhs"] = round(clamped_p / 1e5, 2)
        scatter_samples.append(sample_item)

    metrics_data["scatter_samples"] = scatter_samples

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Updated {metrics_file} with {model_key} successfully!")
