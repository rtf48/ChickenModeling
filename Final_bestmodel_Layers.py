"""
Batch export of best GB models for all fatty-acid deposition outputs
Usage:
    python best_model_export_all.py
"""

import os, joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import Final_listStorage_Layer as ls
import Final_dataset as dataset

BEST_PARAMS_GB = {
    'subsample': 0.6,
    'n_estimators': 300,
    'max_depth': 2,
    'learning_rate': 0.03,
    'random_state': 42
}

OUT_DIR = "outputs/best_models"
os.makedirs(OUT_DIR, exist_ok=True)

def sanitize(s: str) -> str:
    return s.replace('/', '_').replace(' ', '_').replace(':', '_')

def train_and_save(target):
    features = ls.target_features_comp
    X, y = dataset.get_data(features, target)
    if X.shape[0] < 10:
        print(f"[SKIP] Too few samples for {target} ({X.shape[0]})")
        return

    X = X.fillna(X.mean(numeric_only=True))
    model = GradientBoostingRegressor(**BEST_PARAMS_GB)
    model.fit(X, y)

    bundle = {"model": model, "features": list(X.columns)}
    out_path = os.path.join(OUT_DIR, f"{sanitize(target)}__gb.pkl")
    joblib.dump(bundle, out_path)

    print(f"[OK] {target:<25} → {out_path} (n={X.shape[0]}, p={X.shape[1]})")

def main():
    targets = ls.target_labels_5   # all fatty-acid outputs
    for t in targets:
        train_and_save(t)

    print("\n✅ ALL MODELS SAVED TO:", OUT_DIR)

if __name__ == "__main__":
    main()
