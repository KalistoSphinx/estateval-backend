"""
Training & Hyperparameter Tuning: Linear Regression (Ridge Regularized)
Standalone execution: python train_linear_regression.py
"""

import time
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from training_utils import (
    load_data,
    get_preprocessor_scaled,
    evaluate_model_pipeline,
    update_metrics_store,
    map_to_base_feature,
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS
)

def train_linear_regression():
    print("=" * 60)
    print("Training: Linear Regression (Ridge Regularized)")
    print("=" * 60)

    df, X_train, X_test, y_train, y_test, y_test_tiers = load_data()
    preprocessor = get_preprocessor_scaled()

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('reg', Ridge())
    ])

    param_grid = {
        'reg__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'reg__fit_intercept': [True],
        'reg__solver': ['auto', 'svd', 'lsqr']
    }

    print("Running 5-Fold Cross-Validation Grid Search...")
    t0 = time.time()
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    best_pipeline = grid.best_estimator_
    best_params = {k.replace('reg__', ''): v for k, v in grid.best_params_.items()}
    print(f"Best parameters: {best_params}")
    print(f"Training time: {train_time:.2f}s")

    # Evaluate model
    eval_dict, _ = evaluate_model_pipeline(
        model_key="linear_regression",
        model_name="Linear Regression (Ridge Regularized)",
        short_name="Linear Regression",
        pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test,
        y_test_tiers=y_test_tiers,
        train_time=train_time,
        best_params=best_params,
        algorithm_type="Linear / Ridge L2 Regularization",
        tuning_summary="5-Fold Cross Validation optimizing Ridge alpha regularization penalty & solver."
    )

    # Feature Importance: Grouped Absolute Coefficients by Normal Attribute
    cat_encoder = best_pipeline.named_steps['prep'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    all_names = NUMERICAL_COLS + cat_names
    raw_coefs = best_pipeline.named_steps['reg'].coef_

    base_dict = {col: 0.0 for col in FEATURE_COLS}
    for name, c in zip(all_names, raw_coefs):
        base = map_to_base_feature(name)
        base_dict[base] = base_dict.get(base, 0.0) + abs(float(c))

    tot_val = sum(base_dict.values()) or 1.0
    feature_ranking = [
        {"feature": col, "importance": round(val, 2), "percentage": round((val / tot_val) * 100.0, 2)}
        for col, val in sorted(base_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    print(f"Test R2 Score: {eval_dict['metrics']['r2_score'] * 100.0:.2f}%")
    print(f"Test MAE: +/- Rs {eval_dict['metrics']['mae'] / 1e5:.2f} Lakhs")

    update_metrics_store(
        model_key="linear_regression",
        eval_dict=eval_dict,
        feature_ranking=feature_ranking,
        model_pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test
    )
    return best_pipeline, eval_dict

if __name__ == "__main__":
    train_linear_regression()
