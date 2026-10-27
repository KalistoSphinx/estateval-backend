"""
Entry point for training all models.
Delegates to train_all.py or individual scripts:
- train_linear_regression.py
- train_knn.py
- train_random_forest.py
- train_decision_tree.py
"""

from train_all import train_all_models

if __name__ == "__main__":
    train_all_models()
