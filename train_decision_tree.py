"""
Training & Hyperparameter Tuning: Decision Tree Regressor
Standalone execution: python train_decision_tree.py
"""

import time
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

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

def train_decision_tree():
    print("=" * 60)
    print("Training: Decision Tree Regressor")
    print("=" * 60)

    df, X_train, X_test, y_train, y_test, y_test_tiers = load_data()
    preprocessor = get_preprocessor_tree()

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('dt', DecisionTreeRegressor(random_state=42))
    ])

    param_grid = {
        'dt__max_depth': [8, 10, 12, 15, 20, None],
        'dt__min_samples_split': [2, 5, 10, 15],
        'dt__min_samples_leaf': [1, 2, 4, 8],
        'dt__criterion': ['squared_error', 'friedman_mse']
    }

    print("Running 4-Fold Cross-Validation Grid Search for Decision Tree...")
    t0 = time.time()
    grid = GridSearchCV(pipeline, param_grid, cv=4, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    best_pipeline = grid.best_estimator_
    best_params = {k.replace('dt__', ''): v for k, v in grid.best_params_.items()}
    print(f"Best parameters: {best_params}")
    print(f"Training time: {train_time:.2f}s")

    # Evaluate model
    eval_dict, _ = evaluate_model_pipeline(
        model_key="decision_tree",
        model_name="Decision Tree Regressor",
        short_name="Decision Tree",
        pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test,
        y_test_tiers=y_test_tiers,
        train_time=train_time,
        best_params=best_params,
        algorithm_type="Single CART Regression Tree",
        tuning_summary="4-Fold CV optimizing tree depth, minimum split samples, leaf constraints, and split criteria."
    )

    # Feature Importance: Grouped Gini Impurity Reductions by Normal Attribute
    cat_encoder = best_pipeline.named_steps['prep'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    all_names = cat_names + NUMERICAL_COLS
    raw_imps = best_pipeline.named_steps['dt'].feature_importances_

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
        model_key="decision_tree",
        eval_dict=eval_dict,
        feature_ranking=feature_ranking,
        model_pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test
    )
    return best_pipeline, eval_dict

if __name__ == "__main__":
    train_decision_tree()
