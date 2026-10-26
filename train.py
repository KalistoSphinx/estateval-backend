"""
House Price Prediction Model Training & Hyperparameter Tuning
Dataset: Indian Metro House Prices (12,000 properties across Hyderabad, Bangalore, Pune, Mumbai, Nagpur)
Models:
1. Linear Regression (with Ridge regularizer tuning to combat multicollinearity)
2. K-Nearest Neighbors (KNN Regressor with neighbor count, metric, and distance weighting tuning)
3. Random Forest Regressor (with tree count, max depth, split parameters, and feature sampling tuning)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
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
    accuracy_score
)
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "src" / "assets" / "house_price_dataset_india.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PRICE_TIERS = [
    {"name": "Budget (< ₹60L)", "min": 0, "max": 6_000_000},
    {"name": "Mid-Range (₹60L–₹1Cr)", "min": 6_000_000, "max": 10_000_000},
    {"name": "Premium (₹1Cr–₹1.75Cr)", "min": 10_000_000, "max": 17_500_000},
    {"name": "Luxury (> ₹1.75Cr)", "min": 17_500_000, "max": float("inf")},
]
TIER_LABELS = [t["name"] for t in PRICE_TIERS]

def price_to_tier(prices):
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

def compute_adjusted_r2(r2, n, p):
    if n - p - 1 <= 0:
        return r2
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

def train_and_evaluate():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    feature_cols = [
        'City', 'Locality_Tier', 'BHK', 'Bathrooms', 'Super_Area_sqft', 'Carpet_Area_sqft',
        'Floor_No', 'Total_Floors', 'Property_Age_years', 'Parking', 'Furnishing',
        'Lift', 'Gated_Society', 'Distance_to_Metro_km', 'Distance_to_CityCenter_km',
        'Nearby_School_km', 'Nearby_Hospital_km', 'Crime_Rate_Index'
    ]
    target_col = 'Market_Price_INR'

    categorical_cols = ['City', 'Locality_Tier', 'Furnishing']
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Compute metadata and dataset summary
    feature_metadata = {
        "features": feature_cols,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "cities": sorted(df['City'].unique().tolist()),
        "locality_tiers": ['Budget', 'Mid', 'Premium'],
        "furnishing_options": ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'],
        "ranges": {},
        "defaults": {
            "City": "Bangalore",
            "Locality_Tier": "Mid",
            "BHK": 2,
            "Bathrooms": 2,
            "Super_Area_sqft": 1050.0,
            "Carpet_Area_sqft": 820.0,
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
            "Crime_Rate_Index": 35.0,
        }
    }

    for col in numerical_cols:
        feature_metadata["ranges"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }

    dataset_summary = {
        "total_records": len(df),
        "target_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "median": float(y.median()),
            "std": float(y.std())
        },
        "city_distribution": df['City'].value_counts().to_dict(),
        "city_avg_prices": {k: float(v) for k, v in df.groupby('City')['Market_Price_INR'].mean().items()},
        "locality_distribution": df['Locality_Tier'].value_counts().to_dict(),
        "furnishing_distribution": df['Furnishing'].value_counts().to_dict(),
        "bhk_distribution": {str(k): int(v) for k, v in df['BHK'].value_counts().sort_index().items()},
        "tier_counts": {k: int(v) for k, v in pd.Series(price_to_tier(y)).value_counts().items()}
    }

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    y_test_tiers = price_to_tier(y_test.values)

    # Preprocessors
    # 1. Standard scaled + One-Hot encoded for Linear Regression & KNN
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    # 2. Passthrough numerical + One-Hot encoded for Random Forest
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    models_config = {}
    evaluation_results = {}

    # ==========================================
    # 1. LINEAR REGRESSION (with Ridge tuning)
    # ==========================================
    print("\n--- [1/3] Training & Tuning Linear Regression (Ridge) ---")
    start_t = time.time()
    lr_pipeline = Pipeline([
        ('prep', preprocessor_scaled),
        ('reg', Ridge())
    ])
    lr_param_grid = {
        'reg__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'reg__fit_intercept': [True],
        'reg__solver': ['auto', 'svd', 'lsqr']
    }
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='r2', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    lr_fit_time = time.time() - start_t
    best_lr = lr_grid.best_estimator_

    print(f"Linear Regression Best Params: {lr_grid.best_params_}")
    print(f"Linear Regression Train Time: {lr_fit_time:.2f}s")

    # ==========================================
    # 2. KNN REGRESSOR (Hyperparameter tuning)
    # ==========================================
    print("\n--- [2/3] Training & Tuning K-Nearest Neighbors ---")
    start_t = time.time()
    knn_pipeline = Pipeline([
        ('prep', preprocessor_scaled),
        ('knn', KNeighborsRegressor())
    ])
    knn_param_grid = {
        'knn__n_neighbors': [5, 7, 9, 11, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2] # 1=Manhattan, 2=Euclidean
    }
    knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=4, scoring='r2', n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    knn_fit_time = time.time() - start_t
    best_knn = knn_grid.best_estimator_

    print(f"KNN Best Params: {knn_grid.best_params_}")
    print(f"KNN Train Time: {knn_fit_time:.2f}s")

    # ==========================================
    # 3. RANDOM FOREST REGRESSOR (Hyperparameter tuning)
    # ==========================================
    print("\n--- [3/3] Training & Tuning Random Forest Regressor ---")
    start_t = time.time()
    rf_pipeline = Pipeline([
        ('prep', preprocessor_tree),
        ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    rf_param_dist = {
        'rf__n_estimators': [100, 150, 200],
        'rf__max_depth': [20, 25, 30, None],
        'rf__min_samples_split': [2, 4],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 1.0]
    }
    rf_search = RandomizedSearchCV(
        rf_pipeline,
        rf_param_dist,
        n_iter=10,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    rf_fit_time = time.time() - start_t
    best_rf = rf_search.best_estimator_

    print(f"Random Forest Best Params: {rf_search.best_params_}")
    print(f"Random Forest Train Time: {rf_fit_time:.2f}s")

    # Save trained model pipelines
    print("\nSaving trained models to disk...")
    joblib.dump(best_lr, MODELS_DIR / "linear_regression.joblib")
    joblib.dump(best_knn, MODELS_DIR / "knn.joblib")
    joblib.dump(best_rf, MODELS_DIR / "random_forest.joblib")

    # Also fit a baseline OLS LinearRegression for direct comparison
    baseline_lr = Pipeline([
        ('prep', preprocessor_scaled),
        ('reg', LinearRegression())
    ])
    baseline_lr.fit(X_train, y_train)

    trained_models = {
        "linear_regression": {
            "name": "Linear Regression (Ridge Regularized)",
            "short_name": "Linear Regression",
            "model": best_lr,
            "train_time": lr_fit_time,
            "best_params": {k.replace("reg__", ""): v for k, v in lr_grid.best_params_.items()},
            "algorithm_type": "Linear / Ridge L2 Regularization",
            "tuning_summary": "5-Fold Cross Validation optimizing Ridge alpha regularization penalty & solver."
        },
        "knn": {
            "name": "K-Nearest Neighbors Regressor",
            "short_name": "KNN Regressor",
            "model": best_knn,
            "train_time": knn_fit_time,
            "best_params": {k.replace("knn__", ""): v for k, v in knn_grid.best_params_.items()},
            "algorithm_type": "Instance-based Non-Parametric",
            "tuning_summary": "4-Fold CV tuning neighbor count (k=5..15), distance weighting, and Manhattan/Euclidean metrics."
        },
        "random_forest": {
            "name": "Random Forest Regressor",
            "short_name": "Random Forest",
            "model": best_rf,
            "train_time": rf_fit_time,
            "best_params": {k.replace("rf__", ""): v for k, v in rf_search.best_params_.items()},
            "algorithm_type": "Ensemble Bagging of Decision Trees",
            "tuning_summary": "Randomized Search CV optimizing estimators, max depth, leaf samples, and feature subspace sampling."
        }
    }

    # Evaluate all models
    test_samples_count = len(X_test)
    num_features = len(feature_cols)

    # Feature names for importance / coefficient extraction
    # Scaled transformer feature names
    cat_encoder = best_lr.named_steps['prep'].named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names_scaled = numerical_cols + cat_feature_names

    # Tree transformer feature names
    tree_cat_encoder = best_rf.named_steps['prep'].named_transformers_['cat']
    tree_cat_feature_names = tree_cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names_tree = tree_cat_feature_names + numerical_cols

    # Helper to map one-hot and raw features back to base normal attributes
    def map_to_base_feature(name):
        for cat in categorical_cols:
            if name == cat or name.startswith(cat + "_"):
                return cat
        return name

    # 1. Random Forest - Aggregate Gini Impurity Reduction by Base Attribute
    rf_raw_importances = best_rf.named_steps['rf'].feature_importances_
    rf_base_dict = {col: 0.0 for col in feature_cols}
    for name, imp in zip(all_feature_names_tree, rf_raw_importances):
        base = map_to_base_feature(name)
        rf_base_dict[base] = rf_base_dict.get(base, 0.0) + float(imp)
    rf_total = sum(rf_base_dict.values()) or 1.0
    rf_feature_ranking = [
        {"feature": col, "importance": round(val, 4), "percentage": round((val / rf_total) * 100, 2)}
        for col, val in sorted(rf_base_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    # 2. Linear Regression (Ridge) - Aggregate Normalized Absolute Coefficient Weight by Base Attribute
    lr_raw_coefs = best_lr.named_steps['reg'].coef_
    lr_base_dict = {col: 0.0 for col in feature_cols}
    for name, c in zip(all_feature_names_scaled, lr_raw_coefs):
        base = map_to_base_feature(name)
        lr_base_dict[base] = lr_base_dict.get(base, 0.0) + abs(float(c))
    lr_total = sum(lr_base_dict.values()) or 1.0
    lr_feature_ranking = [
        {"feature": col, "importance": round(val, 2), "percentage": round((val / lr_total) * 100, 2)}
        for col, val in sorted(lr_base_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    # 3. KNN - Permutation Feature Importance on Base Attributes
    print("Computing permutation importance for KNN...")
    knn_perm = permutation_importance(best_knn, X_test, y_test, n_repeats=5, random_state=42, scoring='r2', n_jobs=-1)
    knn_imp_mean = np.maximum(0, knn_perm.importances_mean)
    knn_total = float(np.sum(knn_imp_mean)) or 1.0
    knn_pcts = (knn_imp_mean / knn_total) * 100
    knn_feature_ranking = [
        {"feature": name, "importance": round(float(knn_imp_mean[i]), 4), "percentage": round(float(knn_pcts[i]), 2)}
        for i, name in sorted(enumerate(feature_cols), key=lambda x: knn_pcts[x[0]], reverse=True)
    ]

    # Prepare actual vs predicted sample points (100 representative test samples)
    sample_indices = np.linspace(0, len(X_test) - 1, 120, dtype=int)
    sample_actual = y_test.iloc[sample_indices].tolist()

    all_predictions = {}
    scatter_samples = []

    for key, info in trained_models.items():
        model = info["model"]
        
        # Measure inference latency
        t0 = time.perf_counter()
        preds = model.predict(X_test)
        latency_ms = (time.perf_counter() - t0) * 1000 / len(X_test)
        all_predictions[key] = preds

        # Regression Metrics
        r2 = float(r2_score(y_test, preds))
        adj_r2 = float(compute_adjusted_r2(r2, test_samples_count, num_features))
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        mape = float(mean_absolute_percentage_error(y_test, preds) * 100)
        med_ae = float(median_absolute_error(y_test, preds))
        m_err = float(max_error(y_test, preds))
        exp_var = float(explained_variance_score(y_test, preds))

        # Confusion Matrix for Binned Price Tiers
        pred_tiers = price_to_tier(preds)
        cm = confusion_matrix(y_test_tiers, pred_tiers, labels=TIER_LABELS)
        cm_accuracy = float(accuracy_score(y_test_tiers, pred_tiers))

        # Per-tier classification breakdown
        tier_report = classification_report(
            y_test_tiers, pred_tiers, labels=TIER_LABELS, output_dict=True, zero_division=0
        )
        per_tier_metrics = {}
        for t in TIER_LABELS:
            per_tier_metrics[t] = {
                "precision": round(float(tier_report[t]["precision"]), 3),
                "recall": round(float(tier_report[t]["recall"]), 3),
                "f1_score": round(float(tier_report[t]["f1-score"]), 3),
                "support": int(tier_report[t]["support"])
            }

        # Format confusion matrix for UI display
        cm_matrix = []
        for r_idx, actual_label in enumerate(TIER_LABELS):
            row_items = []
            for c_idx, pred_label in enumerate(TIER_LABELS):
                row_items.append({
                    "actual": actual_label,
                    "predicted": pred_label,
                    "count": int(cm[r_idx][c_idx])
                })
            cm_matrix.append(row_items)

        # Residuals
        residuals = y_test.values - preds
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        
        # Residual histogram bins (15 bins)
        hist, bin_edges = np.histogram(residuals / 1e5, bins=15) # in Lakhs
        res_hist = [
            {
                "bin_start": round(float(bin_edges[i]), 1),
                "bin_end": round(float(bin_edges[i+1]), 1),
                "count": int(hist[i])
            }
            for i in range(len(hist))
        ]

        # Error percentiles
        abs_errors = np.abs(residuals)
        percentiles = {
            "p50": float(np.percentile(abs_errors, 50)),
            "p75": float(np.percentile(abs_errors, 75)),
            "p90": float(np.percentile(abs_errors, 90)),
            "p95": float(np.percentile(abs_errors, 95)),
            "p99": float(np.percentile(abs_errors, 99))
        }

        # City-wise error breakdown
        city_test = X_test['City'].values
        city_errors = {}
        for city in feature_metadata["cities"]:
            c_mask = (city_test == city)
            if np.sum(c_mask) > 0:
                city_errors[city] = {
                    "count": int(np.sum(c_mask)),
                    "mae": round(float(mean_absolute_error(y_test[c_mask], preds[c_mask])), 0),
                    "rmse": round(float(root_mean_squared_error(y_test[c_mask], preds[c_mask])), 0),
                    "r2": round(float(r2_score(y_test[c_mask], preds[c_mask])), 4)
                }

        evaluation_results[key] = {
            "name": info["name"],
            "short_name": info["short_name"],
            "algorithm_type": info["algorithm_type"],
            "train_time_sec": round(info["train_time"], 2),
            "latency_ms_per_prediction": round(latency_ms, 3),
            "best_hyperparameters": info["best_params"],
            "tuning_summary": info["tuning_summary"],
            "metrics": {
                "r2_score": round(r2, 4),
                "adjusted_r2": round(adj_r2, 4),
                "mae": round(mae, 0),
                "rmse": round(rmse, 0),
                "mape_percent": round(mape, 2),
                "median_absolute_error": round(med_ae, 0),
                "max_error": round(m_err, 0),
                "explained_variance": round(exp_var, 4)
            },
            "confusion_matrix": {
                "tier_accuracy": round(cm_accuracy * 100, 2),
                "labels": TIER_LABELS,
                "matrix": cm_matrix,
                "raw_matrix": cm.tolist(),
                "tier_breakdown": per_tier_metrics
            },
            "residuals": {
                "mean": round(res_mean, 2),
                "std": round(res_std, 2),
                "histogram": res_hist,
                "percentiles": percentiles
            },
            "city_errors": city_errors
        }

    # Build scatter sample data
    for idx in range(len(sample_indices)):
        actual_val = float(sample_actual[idx])
        scatter_samples.append({
            "id": idx + 1,
            "actual": actual_val,
            "actual_lakhs": round(actual_val / 1e5, 2),
            "linear_regression": float(all_predictions["linear_regression"][sample_indices[idx]]),
            "linear_regression_lakhs": round(float(all_predictions["linear_regression"][sample_indices[idx]]) / 1e5, 2),
            "knn": float(all_predictions["knn"][sample_indices[idx]]),
            "knn_lakhs": round(float(all_predictions["knn"][sample_indices[idx]]) / 1e5, 2),
            "random_forest": float(all_predictions["random_forest"][sample_indices[idx]]),
            "random_forest_lakhs": round(float(all_predictions["random_forest"][sample_indices[idx]]) / 1e5, 2),
        })

    # Combine full metrics payload
    full_metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_summary": dataset_summary,
        "models": evaluation_results,
        "feature_importances": {
            "random_forest": rf_feature_ranking[:12],
            "linear_regression": lr_feature_ranking[:12],
            "knn": knn_feature_ranking[:12]
        },
        "scatter_samples": scatter_samples,
        "tier_labels": TIER_LABELS,
        "best_overall_model": "random_forest"
    }

    # Save metrics JSON & metadata JSON
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    with open(MODELS_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=2)

    print("\nTraining, hyperparameter tuning & evaluation completed successfully!")
    print(f"Metrics saved to: {MODELS_DIR / 'metrics.json'}")
    print(f"Feature metadata saved to: {MODELS_DIR / 'feature_metadata.json'}")
    return full_metrics

if __name__ == "__main__":
    train_and_evaluate()
