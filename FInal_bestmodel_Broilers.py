"""
Batch export best GB models for broiler fatty-acid outputs
Run:
    python best_model_export_all_broiler.py
"""

import os, joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import Final_listStorage_Broiler as ls   # ✅ IMPORT BROILER LIST
import Final_dataset as dataset

BEST_PARAMS_GB = {
    'subsample': 0.6,
    'n_estimators': 300,
    'max_depth': 2,
    'learning_rate': 0.03,
    'random_state': 42
}

OUT_DIR = "outputs/best_models_broiler"
os.makedirs(OUT_DIR, exist_ok=True)

def sanitize(s: str) -> str:
    return (
        s.replace('/', '_')
        .replace(' ', '_')
        .replace(':', '_')
        .replace(',', '_')
        .replace('(', '_')
        .replace(')', '_')
    )

def train_and_save(target):
    features = ls.target_features_comp
    X, y = dataset.get_data(features, target)

    # skip if too few samples
    if X.shape[0] < 10:
        print(f"[SKIP] Too few samples for {target} (n={X.shape[0]})")
        return

    # fill NA
    X = X.fillna(X.mean(numeric_only=True))

    # fit model
    model = GradientBoostingRegressor(**BEST_PARAMS_GB)
    model.fit(X, y)

    # save bundle
    bundle = {"model": model, "features": list(X.columns)}
    out_path = os.path.join(OUT_DIR, f"{sanitize(target)}__gb.pkl")
    joblib.dump(bundle, out_path)

    print(f"[OK] {target:<25} → {out_path} (n={X.shape[0]}, p={X.shape[1]})")

def main():
    targets = ls.target_labels_5   # ✅ Broiler FA list
    print(f"Exporting {len(targets)} broiler FA models...")
    for t in targets:
        train_and_save(t)

    print(f"\n✅ ALL BROILER MODELS SAVED TO: {OUT_DIR}")

if __name__ == "__main__":
    main()
