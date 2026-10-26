import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from schemas import HouseFeatures, ModelPrediction, MultiPredictionResponse
except ImportError:
    from backend.schemas import HouseFeatures, ModelPrediction, MultiPredictionResponse

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="House Price Valuation & Model Benchmarking API",
    description="FastAPI service serving tuned Linear Regression, KNN, and Random Forest models on Indian Metro real estate data.",
    version="1.0.0"
)

# Enable CORS for React frontend (local dev & production Vercel app)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://estate-val.vercel.app",
    "https://estate-val.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app/?.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model store
models: Dict[str, Any] = {}
metrics_data: Optional[Dict[str, Any]] = None
feature_metadata: Optional[Dict[str, Any]] = None

def format_indian_currency(amount: float) -> str:
    """Formats an amount in INR with Lakhs and Crores notations."""
    if amount < 0:
        return f"-₹{abs(amount):,.0f}"
    if amount >= 10_000_000:
        crores = amount / 10_000_000
        return f"₹{crores:.2f} Cr"
    elif amount >= 100_000:
        lakhs = amount / 100_000
        return f"₹{lakhs:.2f} Lakh"
    else:
        return f"₹{amount:,.0f}"

def get_tier(price: float) -> str:
    if price < 6_000_000:
        return "Budget (< ₹60L)"
    elif price < 10_000_000:
        return "Mid-Range (₹60L–₹1Cr)"
    elif price < 17_500_000:
        return "Premium (₹1Cr–₹1.75Cr)"
    else:
        return "Luxury (> ₹1.75Cr)"

@app.on_event("startup")
def load_models_and_metadata():
    global models, metrics_data, feature_metadata
    print("Loading models and metadata into memory...")

    # Load metrics
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        print("Loaded model metrics.")
    else:
        print("Warning: metrics.json not found. Run train.py first.")

    # Load feature metadata
    meta_path = MODELS_DIR / "feature_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            feature_metadata = json.load(f)
        print("Loaded feature metadata.")

    # Load scikit-learn models
    model_files = {
        "linear_regression": "linear_regression.joblib",
        "knn": "knn.joblib",
        "random_forest": "random_forest.joblib"
    }

    for key, filename in model_files.items():
        file_path = MODELS_DIR / filename
        if file_path.exists():
            models[key] = joblib.load(file_path)
            print(f"Loaded {key} from {filename}")
        else:
            print(f"Warning: {filename} not found.")

@app.get("/")
def read_root():
    return {
        "service": "House Price Valuation API",
        "status": "online",
        "loaded_models": list(models.keys()),
        "endpoints": {
            "metrics": "/api/models/metrics",
            "features": "/api/features",
            "predict": "/api/predict",
            "dataset_summary": "/api/dataset/summary",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy" if len(models) == 3 else "degraded",
        "models_loaded": {k: (k in models) for k in ["linear_regression", "knn", "random_forest"]},
        "metrics_available": metrics_data is not None
    }

@app.get("/api/features")
def get_features():
    if not feature_metadata:
        raise HTTPException(status_code=500, detail="Feature metadata is not loaded.")
    return feature_metadata

@app.get("/api/models/metrics")
def get_model_metrics():
    global metrics_data
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    if not metrics_data:
        raise HTTPException(status_code=500, detail="Metrics are not loaded.")
    return metrics_data

@app.get("/api/dataset/summary")
def get_dataset_summary():
    if not metrics_data or "dataset_summary" not in metrics_data:
        raise HTTPException(status_code=500, detail="Dataset summary not found.")
    return metrics_data["dataset_summary"]

@app.post("/api/predict", response_model=MultiPredictionResponse)
def predict_price(features: HouseFeatures):
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded in memory.")

    # Convert features to DataFrame
    input_dict = {
        'City': [features.City],
        'Locality_Tier': [features.Locality_Tier],
        'BHK': [features.BHK],
        'Bathrooms': [features.Bathrooms],
        'Super_Area_sqft': [features.Super_Area_sqft],
        'Carpet_Area_sqft': [features.Carpet_Area_sqft],
        'Floor_No': [features.Floor_No],
        'Total_Floors': [features.Total_Floors],
        'Property_Age_years': [features.Property_Age_years],
        'Parking': [features.Parking],
        'Furnishing': [features.Furnishing],
        'Lift': [features.Lift],
        'Gated_Society': [features.Gated_Society],
        'Distance_to_Metro_km': [features.Distance_to_Metro_km],
        'Distance_to_CityCenter_km': [features.Distance_to_CityCenter_km],
        'Nearby_School_km': [features.Nearby_School_km],
        'Nearby_Hospital_km': [features.Nearby_Hospital_km],
        'Crime_Rate_Index': [features.Crime_Rate_Index]
    }
    input_df = pd.DataFrame(input_dict)

    predictions = {}
    target_models = (
        [features.model]
        if features.model and features.model in models
        else ["linear_regression", "knn", "random_forest"]
    )

    prices_list = []
    display_names = {
        "linear_regression": "Linear Regression (Ridge)",
        "knn": "K-Nearest Neighbors",
        "random_forest": "Random Forest Regressor"
    }

    for m_key in target_models:
        if m_key not in models:
            continue
        model = models[m_key]
        t0 = time.perf_counter()
        raw_pred = float(model.predict(input_df)[0])
        latency_ms = (time.perf_counter() - t0) * 1000

        # Sanity floor to avoid negative predictions from unbounded linear models on extreme edges
        pred_val = max(100_000.0, raw_pred)
        prices_list.append(pred_val)

        price_per_sqft = pred_val / max(1.0, features.Super_Area_sqft)

        predictions[m_key] = ModelPrediction(
            model_key=m_key,
            model_name=display_names.get(m_key, m_key),
            predicted_price_inr=round(pred_val, 2),
            price_in_lakhs=round(pred_val / 100_000, 2),
            price_in_crores=round(pred_val / 10_000_000, 3),
            formatted_inr=format_indian_currency(pred_val),
            price_per_sqft=round(price_per_sqft, 2),
            price_tier=get_tier(pred_val),
            latency_ms=round(latency_ms, 2)
        )

    ensemble_avg = float(np.mean(prices_list)) if prices_list else 0.0
    price_spread = 0.0
    if len(prices_list) > 1 and min(prices_list) > 0:
        price_spread = ((max(prices_list) - min(prices_list)) / ensemble_avg) * 100

    return MultiPredictionResponse(
        input_summary={
            "City": features.City,
            "Locality": features.Locality_Tier,
            "BHK": features.BHK,
            "Super_Area_sqft": features.Super_Area_sqft,
            "Furnishing": features.Furnishing
        },
        predictions=predictions,
        ensemble_average_inr=round(ensemble_avg, 2),
        ensemble_average_formatted=format_indian_currency(ensemble_avg),
        price_spread_percentage=round(price_spread, 2),
        recommended_model="random_forest",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
