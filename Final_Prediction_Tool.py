from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

# =========================
# paths
# =========================
PKL_PATH = "outputs/layer_DHA_Final.pkl"
REF_CSV = "Individual_Layers.csv"
ING_CSV = "Final_Oil_Ing.csv"

# =========================
# BD formula (g/kg diet)
# =========================
BD_FORMULA = {
    "Corn grain (dry)": 651.0,
    "Soybean meal": 237.9,
    "Corn oil": 15.0,
}

START = 168
END = 195

# ingredient-library columns -> model columns
FA_MAP = {
    "C14": "C14,g",
    "C16:0": "C16:0,g",
    "C16:1": "C16:1,g",
    "C18:0": "C18:0,g",
    "C18:1": "C18:1,g",
    "C18:2 n-6": "C18:2 cis n-6 LA,g",
    "C18:3 n-3": "C18:3 cis n-3 ALA,g",
    "C20:4 n-6": "C20:4n-6 ARA,g",
    "C20:5 n-3": "C20:5n-3 EPA,g",
    "C22:6 n-3": "C22:6n-3 DHA,g",
}


def compute_bd_fa(ing_csv: str, formula_g_per_kg: dict[str, float]) -> dict[str, float]:
    df = pd.read_csv(ing_csv)
    df.columns = df.columns.str.strip()

    if "Ingredient" not in df.columns:
        raise KeyError("Ingredient library must contain column 'Ingredient'.")

    df["Ingredient_norm"] = df["Ingredient"].astype(str).str.strip().str.lower()

    # initialize FA totals
    bd_fa = {model_col: 0.0 for model_col in FA_MAP.values()}

    for ing_name, inclusion_g in formula_g_per_kg.items():
        match = df[df["Ingredient_norm"] == ing_name.strip().lower()]
        if match.empty:
            raise ValueError(f"Ingredient not found in library: {ing_name}")

        row = match.iloc[0]
        frac = inclusion_g / 1000.0

        for ing_col, model_col in FA_MAP.items():
            value = float(row[ing_col])
            bd_fa[model_col] += value * frac

    return bd_fa


def main() -> None:
    # 1) compute BD FA profile from ingredient library
    bd_fa = compute_bd_fa(ING_CSV, BD_FORMULA)

    # 2) load model bundle
    bundle = joblib.load(PKL_PATH)
    model = bundle["model"]
    features = list(bundle["features"])
    is_log = bundle.get("log_transformed", False)

    # 3) load reference data for median defaults
    df_raw = pd.read_csv(REF_CSV)
    missing = [c for c in features if c not in df_raw.columns]
    if missing:
        raise KeyError(
            "Reference CSV is missing required model features:\n" + "\n".join(missing)
        )

    df_ref = df_raw[features].copy()
    for c in features:
        df_ref[c] = pd.to_numeric(df_ref[c], errors="coerce")

    base_vec = df_ref.median(numeric_only=True)

    # 4) build model input row
    x = pd.DataFrame([base_vec[features].values], columns=features)

    if "Start" in x.columns:
        x.loc[0, "Start"] = START
    if "End" in x.columns:
        x.loc[0, "End"] = END
    if "Window Length" in x.columns:
        x.loc[0, "Window Length"] = END - START + 1

    for k, v in bd_fa.items():
        if k in x.columns:
            x.loc[0, k] = v

    # 5) base prediction (median baseline)
    base = pd.DataFrame([base_vec[features].values], columns=features)
    if "Start" in base.columns:
        base.loc[0, "Start"] = START
    if "End" in base.columns:
        base.loc[0, "End"] = END
    if "Window Length" in base.columns:
        base.loc[0, "Window Length"] = END - START + 1

    raw_base = float(model.predict(base[features])[0])
    base_pred = np.expm1(raw_base) if is_log else raw_base

    # 6) predict BD
    raw_pred = float(model.predict(x[features])[0])
    pred = np.expm1(raw_pred) if is_log else raw_pred

    improve_pct = (pred - base_pred) / (base_pred + 1e-12) * 100.0

    print("=== Computed BD fatty acid profile ===")
    for k, v in bd_fa.items():
        print(f"{k}: {v:.10f}")

    print("\n=== Model prediction ===")
    print(f"Base prediction: {base_pred:.6f}")
    print(f"BD prediction:   {pred:.6f}")
    print(f"Improve % vs base: {improve_pct:.2f}%")


if __name__ == "__main__":
    main()