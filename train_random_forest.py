"""
Training & Hyperparameter Tuning: Random Forest Regressor
Standalone execution: python train_random_forest.py
"""

import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from training_utils import (
    load_data,
    get_preprocessor_tree,
    evaluate_model_pipeline,
    update_metrics_store,
    map_to_base_feature,
    FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS
)

def train_random_forest():
    print("=" * 60)
    print("Training: Random Forest Regressor")
    print("=" * 60)

    df, X_train, X_test, y_train, y_test, y_test_tiers = load_data()
    preprocessor = get_preprocessor_tree()

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_dist = {
        'rf__n_estimators': [100, 150, 200],
        'rf__max_depth': [20, 25, 30, None],
        'rf__min_samples_split': [2, 4],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 1.0]
    }

    print("Running Randomized Search Cross-Validation (10 iterations)...")
    t0 = time.time()
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=10,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    train_time = time.time() - t0

    best_pipeline = search.best_estimator_
    best_params = {k.replace('rf__', ''): v for k, v in search.best_params_.items()}
    print(f"Best parameters: {best_params}")
    print(f"Training time: {train_time:.2f}s")

    # Evaluate model
    eval_dict, _ = evaluate_model_pipeline(
        model_key="random_forest",
        model_name="Random Forest Regressor",
        short_name="Random Forest",
        pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test,
        y_test_tiers=y_test_tiers,
        train_time=train_time,
        best_params=best_params,
        algorithm_type="Ensemble Bagging of Decision Trees",
        tuning_summary="Randomized Search CV optimizing estimators, max depth, leaf samples, and feature subspace sampling."
    )

    # Feature Importance: Grouped Gini Impurity Reductions by Normal Attribute
    cat_encoder = best_pipeline.named_steps['prep'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    all_names = cat_names + NUMERICAL_COLS
    raw_imps = best_pipeline.named_steps['rf'].feature_importances_

    base_dict = {col: 0.0 for col in FEATURE_COLS}
    for name, imp in zip(all_names, raw_imps):
        base = map_to_base_feature(name)
        base_dict[base] = base_dict.get(base, 0.0) + float(imp)

    tot_val = sum(base_dict.values()) or 1.0
    feature_ranking = [
        {"feature": col, "importance": round(val, 4), "percentage": round((val / tot_val) * 100.0, 2)}
        for col, val in sorted(base_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    print(f"Test R2 Score: {eval_dict['metrics']['r2_score'] * 100.0:.2f}%")
    print(f"Test MAE: +/- Rs {eval_dict['metrics']['mae'] / 1e5:.2f} Lakhs")

    update_metrics_store(
        model_key="random_forest",
        eval_dict=eval_dict,
        feature_ranking=feature_ranking,
        model_pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test
    )
    return best_pipeline, eval_dict

if __name__ == "__main__":
    train_random_forest()
