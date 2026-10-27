"""
Training & Hyperparameter Tuning: K-Nearest Neighbors (KNN Regressor)
Standalone execution: python train_knn.py
"""

import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from training_utils import (
    load_data,
    get_preprocessor_scaled,
    evaluate_model_pipeline,
    update_metrics_store,
    FEATURE_COLS
)

def train_knn():
    print("=" * 60)
    print("Training: K-Nearest Neighbors Regressor")
    print("=" * 60)

    df, X_train, X_test, y_train, y_test, y_test_tiers = load_data()
    preprocessor = get_preprocessor_scaled()

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('knn', KNeighborsRegressor())
    ])

    param_grid = {
        'knn__n_neighbors': [5, 7, 9, 11, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # 1=Manhattan, 2=Euclidean
    }

    print("Running 4-Fold Cross-Validation Grid Search...")
    t0 = time.time()
    grid = GridSearchCV(pipeline, param_grid, cv=4, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    best_pipeline = grid.best_estimator_
    best_params = {k.replace('knn__', ''): v for k, v in grid.best_params_.items()}
    print(f"Best parameters: {best_params}")
    print(f"Training time: {train_time:.2f}s")

    # Evaluate model
    eval_dict, _ = evaluate_model_pipeline(
        model_key="knn",
        model_name="K-Nearest Neighbors Regressor",
        short_name="KNN Regressor",
        pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test,
        y_test_tiers=y_test_tiers,
        train_time=train_time,
        best_params=best_params,
        algorithm_type="Instance-based Non-Parametric",
        tuning_summary="4-Fold CV tuning neighbor count (k=5..15), distance weighting, and Manhattan/Euclidean metrics."
    )

    # Feature Importance: Permutation Importance on Normal Attributes
    print("Computing Permutation Feature Importance for KNN on holdout set...")
    perm = permutation_importance(best_pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring='r2', n_jobs=-1)
    imp_mean = np.maximum(0, perm.importances_mean)
    imp_total = float(np.sum(imp_mean)) or 1.0
    pcts = (imp_mean / imp_total) * 100.0
    feature_ranking = [
        {"feature": name, "importance": round(float(imp_mean[i]), 4), "percentage": round(float(pcts[i]), 2)}
        for i, name in sorted(enumerate(FEATURE_COLS), key=lambda x: pcts[x[0]], reverse=True)
    ]

    print(f"Test R2 Score: {eval_dict['metrics']['r2_score'] * 100.0:.2f}%")
    print(f"Test MAE: +/- Rs {eval_dict['metrics']['mae'] / 1e5:.2f} Lakhs")

    update_metrics_store(
        model_key="knn",
        eval_dict=eval_dict,
        feature_ranking=feature_ranking,
        model_pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test
    )
    return best_pipeline, eval_dict

if __name__ == "__main__":
    train_knn()
