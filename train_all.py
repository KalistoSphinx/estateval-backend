"""
Master Training Runner: Trains all 4 models sequentially and updates the unified metrics store.
Models:
1. Linear Regression (Ridge regularized)
2. KNN Regressor
3. Random Forest Regressor
4. Decision Tree Regressor
"""

import time
from train_linear_regression import train_linear_regression
from train_knn import train_knn
from train_random_forest import train_random_forest
from train_decision_tree import train_decision_tree

def train_all_models():
    print("=" * 70)
    print("STARTING FULL MODEL TRAINING PIPELINE FOR 4 ALGORITHMS")
    print("=" * 70)
    t_start = time.time()

    # 1. Linear Regression
    print("\n>>> [1/4] Training Linear Regression...")
    lr_model, lr_eval = train_linear_regression()

    # 2. Decision Tree (fast, tree-based)
    print("\n>>> [2/4] Training Decision Tree Regressor...")
    dt_model, dt_eval = train_decision_tree()

    # 3. K-Nearest Neighbors
    print("\n>>> [3/4] Training K-Nearest Neighbors...")
    knn_model, knn_eval = train_knn()

    # 4. Random Forest
    print("\n>>> [4/4] Training Random Forest Regressor...")
    rf_model, rf_eval = train_random_forest()

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ALL 4 MODELS TRAINED SUCCESSFULLY IN {total_time:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    train_all_models()
