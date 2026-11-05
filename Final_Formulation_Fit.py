from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1) USER-STYLE DEFAULTS
# ============================================================
DEFAULT_PKL = "outputs/layer_DHA_Final.pkl"
DEFAULT_REF_CSV = "Individual_Layers.csv"
DEFAULT_INGREDIENTS_CSV = "Final_Oil_Ing.csv"
DEFAULT_OUT_DIR = "outputs/ingredient_space"

SBM_G_PER_1000 = 237.9
FIXED_ADDITIVES_G = 99.1
OIL_SUM_MIN_G = 0.0
OIL_SUM_MAX_G = 100.0

SEED = 42

# IND-FA columns used by the model / reference csv
IND_FA_COLS_MODEL = [
    "C14,g",
    "C16:0,g",
    "C16:1,g",
    "C18:0,g",
    "C18:1,g",
    "C18:2 cis n-6 LA,g",
    "C18:3 cis n-3 ALA,g",
    "C20:4n-6 ARA,g",
    "C20:5n-3 EPA,g",
    "C22:6n-3 DHA,g",
]

# Corresponding ingredient-library columns
IND_FA_COLS_ING = [
    "C14",
    "C16:0",
    "C16:1",
    "C18:0",
    "C18:1",
    "C18:2 n-6",
    "C18:3 n-3",
    "C20:4 n-6",
    "C20:5 n-3",
    "C22:6 n-3"
]


# ============================================================
# 2) UTILITIES
# ============================================================
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def resolve_fa_cols(df_cols: List[str], requested_cols: List[str]) -> List[str]:
    """
    Resolve column names between ',g' and ',g/g diet' styles.
    """
    resolved = []
    for c in requested_cols:
        if c in df_cols:
            resolved.append(c)
            continue

        alt1 = c.replace(",g/g diet", ",g")
        alt2 = c.replace(",g", ",g/g diet")

        if alt1 in df_cols:
            resolved.append(alt1)
        elif alt2 in df_cols:
            resolved.append(alt2)
        else:
            raise KeyError(f"Column not found: {c}")
    return resolved


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived metrics from model-style IND FA columns.
    """
    out = df.copy()

    def get(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(out), dtype=float), index=out.index)

    c14 = get("C14,g")
    c160 = get("C16:0,g")
    c161 = get("C16:1,g")
    c180 = get("C18:0,g")
    c181 = get("C18:1,g")
    la = get("C18:2 cis n-6 LA,g")
    ala = get("C18:3 cis n-3 ALA,g")
    ara = get("C20:4n-6 ARA,g")
    epa = get("C20:5n-3 EPA,g")
    dha = get("C22:6n-3 DHA,g")

    out["SFA_derived,g/g diet"] = c14 + c160 + c180
    out["MUFA_derived,g/g diet"] = c161 + c181
    out["n-3_derived,g/g diet"] = ala + epa + dha
    out["n-6_derived,g/g diet"] = la + ara
    out["PUFA_derived,g/g diet"] = out["n-3_derived,g/g diet"] + out["n-6_derived,g/g diet"]
    out["TotalFat_derived,g/g diet"] = (
        c14 + c160 + c161 + c180 + c181 + la + ala + ara + epa + dha
    )
    return out


def dirichlet_oil_sampling(
    n_samples: int,
    n_oils: int,
    oil_sum_min_g: float,
    oil_sum_max_g: float,
    seed: int,
    alpha: float = 1.0,
    sparse_alpha: bool = False,
) -> np.ndarray:
    """
    Sample oil inclusion levels (g/kg diet) directly.

    Strategy:
    - sample total oil amount T uniformly in [min, max]
    - sample composition p from Dirichlet
    - x = T * p
    """
    rng = np.random.default_rng(seed)

    if sparse_alpha:
        alpha_vec = rng.uniform(0.15, alpha, size=n_oils)
    else:
        alpha_vec = np.full(n_oils, alpha, dtype=float)

    totals = rng.uniform(oil_sum_min_g, oil_sum_max_g, size=n_samples)
    props = rng.dirichlet(alpha_vec, size=n_samples)
    x_oil_g = props * totals[:, None]
    return x_oil_g


def build_fa_from_oils(
    x_oil_g: np.ndarray,
    oil_names: List[str],
    a_ing: np.ndarray,
    corn_fa: np.ndarray,
    sbm_fa: np.ndarray,
    sbm_g_per_1000: float,
    fixed_additives_g: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert sampled oil formulations to achievable IND FA profile.
    """
    x_oil_frac = x_oil_g / 1000.0
    sbm_frac = sbm_g_per_1000 / 1000.0
    fixed_frac = fixed_additives_g / 1000.0

    oil_sum_frac = x_oil_frac.sum(axis=1)
    corn_frac = 1.0 - sbm_frac - fixed_frac - oil_sum_frac

    feasible_mask = corn_frac >= -1e-12
    x_oil_frac = x_oil_frac[feasible_mask]
    corn_frac = corn_frac[feasible_mask]
    x_oil_g = x_oil_g[feasible_mask]

    achieved = (
        sbm_fa[None, :] * sbm_frac
        + corn_fa[None, :] * corn_frac[:, None]
        + x_oil_frac @ a_ing.T
    )

    fa_df = pd.DataFrame(achieved, columns=IND_FA_COLS_MODEL)

    oil_df = pd.DataFrame(x_oil_g, columns=[f"{x}_g_per_kg" for x in oil_names])
    oil_df["Oil_g_per_kg_total"] = x_oil_g.sum(axis=1)
    oil_df["Corn_g_per_kg"] = corn_frac * 1000.0
    oil_df["SBM_g_per_kg"] = sbm_g_per_1000
    oil_df["FixedAdditives_g_per_kg"] = fixed_additives_g

    return fa_df, oil_df


def load_ingredient_library(
    ingredients_csv: str,
    oil_names_keep: List[str] | None = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, List[str], List[str]]:
    """
    Load ingredient library and return:
    - corn row
    - sbm row
    - oils dataframe
    - resolved ingredient FA columns
    - oil names
    """
    df_i = pd.read_csv(ingredients_csv)
    df_i.columns = df_i.columns.str.strip()

    if "Ingredient" not in df_i.columns:
        raise KeyError("Ingredient library must contain column: Ingredient")

    df_i["norm_name"] = df_i["Ingredient"].astype(str).str.lower().str.strip()

    corn_match = df_i[df_i["norm_name"].str.contains("corn grain", na=False)]
    sbm_match = df_i[df_i["norm_name"].str.contains("soybean meal", na=False)]

    if corn_match.empty:
        raise ValueError("No ingredient matched 'corn grain'. Check Ingredient names.")
    if sbm_match.empty:
        raise ValueError("No ingredient matched 'soybean meal'. Check Ingredient names.")

    corn = corn_match.iloc[0]
    sbm = sbm_match.iloc[0]

    fa_cols_ing = resolve_fa_cols(df_i.columns.tolist(), IND_FA_COLS_ING)

    exclude_patterns = ["corn grain", "soybean meal"]
    oils_df = df_i[~df_i["norm_name"].str.contains("|".join(exclude_patterns), na=False)].copy()

    if oil_names_keep:
        keep_norm = [x.lower().strip() for x in oil_names_keep]
        oils_df["_keep"] = oils_df["norm_name"].isin(keep_norm)
        oils_df = oils_df[oils_df["_keep"]].copy()
        oils_df = oils_df.drop(columns="_keep", errors="ignore")

    if oils_df.empty:
        raise ValueError("No candidate oils found after filtering.")

    oil_names = oils_df["Ingredient"].astype(str).tolist()
    return corn, sbm, oils_df, fa_cols_ing, oil_names


def prepare_model_base(
    pkl_path: str,
    ref_csv: str,
    start: float,
    end: float,
) -> Tuple[object, List[str], bool, pd.DataFrame, pd.DataFrame, float]:
    """
    Load locked model and build base prediction.
    """
    bundle = joblib.load(pkl_path)
    model = bundle["model"]
    features = list(bundle["features"])
    is_log = bundle.get("log_transformed", False)

    df_raw = pd.read_csv(ref_csv)
    missing = [c for c in features if c not in df_raw.columns]
    if missing:
        raise KeyError(
            "Reference CSV is missing required model features:\n"
            + "\n".join(missing)
        )

    df_ref = coerce_numeric(df_raw[features], features)

    base_vec = df_ref.median(numeric_only=True)
    base = pd.DataFrame([base_vec[features].values], columns=features)

    if "Start" in base.columns:
        base["Start"] = start
    if "End" in base.columns:
        base["End"] = end
    if "Window Length" in base.columns:
        base["Window Length"] = end - start + 1

    raw_base_pred = float(model.predict(base[features])[0])
    base_pred = np.expm1(raw_base_pred) if is_log else raw_base_pred

    return model, features, is_log, df_raw, df_ref, base_pred


def build_candidate_feature_df(
    fa_df: pd.DataFrame,
    features: List[str],
    df_ref: pd.DataFrame,
    start: float,
    end: float,
) -> pd.DataFrame:
    """
    Build full feature dataframe for prediction.
    """
    base_vec = df_ref.median(numeric_only=True)
    cand = pd.DataFrame(
        np.tile(base_vec[features].values, (len(fa_df), 1)),
        columns=features
    )

    for c in IND_FA_COLS_MODEL:
        if c in cand.columns and c in fa_df.columns:
            cand[c] = fa_df[c].values

    if "Start" in cand.columns:
        cand["Start"] = start
    if "End" in cand.columns:
        cand["End"] = end
    if "Window Length" in cand.columns:
        cand["Window Length"] = end - start + 1

    return cand


def apply_constraints(
    cand: pd.DataFrame,
    df_ref: pd.DataFrame,
    mode: str,
    q_low: float,
    q_high: float,
    eps: float = 1e-12,
    sfa_limit: float = 0.0060,
    mufa_limit: float = 0.0100,
    crude_fat_target: float | None = None,
    crude_fat_tol: float | None = None,
) -> pd.DataFrame:
    """
    mode:
      - free: keep TotalFat_derived within reference quantiles
      - fixed_cf: keep TotalFat_derived within target ± tol
    """
    c = cand.copy()

    if all(k in c.columns for k in ["Start", "End", "Window Length"]):
        c["Window Length"] = c["End"] - c["Start"] + 1

    c = add_derived_metrics(c)

    c = c.loc[c["SFA_derived,g/g diet"] >= sfa_limit].copy()
    c = c.loc[c["MUFA_derived,g/g diet"] >= mufa_limit].copy()

    if len(c) == 0:
        return c.reset_index(drop=True)

    if mode == "fixed_cf":
        if crude_fat_target is None or crude_fat_tol is None:
            raise ValueError(
                "When mode='fixed_cf', both crude_fat_target and crude_fat_tol must be provided."
            )

        lo = crude_fat_target - crude_fat_tol
        hi = crude_fat_target + crude_fat_tol

        c = c.loc[
            (c["TotalFat_derived,g/g diet"] >= lo - eps)
            & (c["TotalFat_derived,g/g diet"] <= hi + eps)
        ].copy()

    elif mode == "free":
        ref = add_derived_metrics(df_ref.copy())
        ref_total = (
            pd.to_numeric(ref["TotalFat_derived,g/g diet"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if not ref_total.empty:
            lo = float(ref_total.quantile(q_low))
            hi = float(ref_total.quantile(q_high))
            c = c.loc[
                (c["TotalFat_derived,g/g diet"] >= lo - eps)
                & (c["TotalFat_derived,g/g diet"] <= hi + eps)
            ].copy()
    else:
        raise ValueError("mode must be one of: ['free', 'fixed_cf']")

    return c.reset_index(drop=True)


def select_two_diverse_below_cap(
    cand_full: pd.DataFrame,
    pred_dha_cap: float = 150.0,
    near_cap_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Select exactly 2 diets:
    - both must satisfy pred_DHA <= pred_dha_cap
    - among eligible diets, prefer those as close as possible to the cap
    - then choose the pair with the largest FA-profile distance within that near-cap pool

    Parameters
    ----------
    pred_dha_cap : float
        Upper limit for predicted DHA.
    near_cap_quantile : float
        Among diets below cap, keep the top fraction closest to the cap.
        Example:
            0.90 means keep the top 10% closest to the cap.
            0.80 means keep the top 20% closest to the cap.
    """
    if len(cand_full) < 2:
        raise RuntimeError("Need at least 2 valid candidate diets.")

    if "pred_DHA" not in cand_full.columns:
        raise KeyError("cand_full must contain column 'pred_DHA' before selection.")

    div_cols = [c for c in IND_FA_COLS_MODEL if c in cand_full.columns]
    if not div_cols:
        raise RuntimeError("No IND FA columns found for diversity selection.")

    # Step 1: keep only diets below or equal to cap
    eligible = cand_full.loc[cand_full["pred_DHA"] <= pred_dha_cap].copy().reset_index(drop=True)

    if len(eligible) < 2:
        raise RuntimeError(
            f"Fewer than 2 candidate diets satisfy pred_DHA <= {pred_dha_cap}. "
            f"Try increasing the cap or relaxing other constraints."
        )

    # Step 2: among eligible diets, prefer those closest to the cap
    # closeness = higher pred_DHA is better, as long as <= cap
    threshold = float(np.quantile(eligible["pred_DHA"], near_cap_quantile))
    near_cap_pool = eligible.loc[eligible["pred_DHA"] >= threshold].copy().reset_index(drop=True)

    # fallback if pool too small
    if len(near_cap_pool) < 2:
        near_cap_pool = eligible.sort_values("pred_DHA", ascending=False).head(min(20, len(eligible))).copy().reset_index(drop=True)

    x = near_cap_pool[div_cols].to_numpy(dtype=float)

    # standardize for distance calculation
    std = x.std(axis=0)
    std[std == 0] = 1.0
    x_scaled = x / std

    best_i = None
    best_j = None
    best_dist = -np.inf

    n = len(x_scaled)
    for i in range(n):
        diff = x_scaled[i + 1:] - x_scaled[i]
        dists = np.sqrt((diff ** 2).sum(axis=1))
        if len(dists) == 0:
            continue
        j_rel = int(np.argmax(dists))
        dist_ij = float(dists[j_rel])
        j = i + 1 + j_rel

        if dist_ij > best_dist:
            best_dist = dist_ij
            best_i, best_j = i, j

    out = near_cap_pool.iloc[[best_i, best_j]].copy().reset_index(drop=True)
    out.insert(0, "SelectedGroup", ["Group_1_below_cap", "Group_2_below_cap"])
    out["selection_note"] = [
        f"Selected under pred_DHA cap <= {pred_dha_cap}, near-cap pool, maximum FA-profile distance",
        f"Selected under pred_DHA cap <= {pred_dha_cap}, near-cap pool, maximum FA-profile distance",
    ]
    return out


def save_outputs(
    out_dir: str,
    stamp: str,
    selected_df: pd.DataFrame,
    all_valid: pd.DataFrame,
) -> Dict[str, str]:
    safe_mkdir(out_dir)

    selected_path = os.path.join(out_dir, f"ingredient_space_selected_{stamp}.csv")
    all_valid_path = os.path.join(out_dir, f"ingredient_space_all_valid_{stamp}.csv")

    selected_df.to_csv(selected_path, index=False)
    all_valid.to_csv(all_valid_path, index=False)

    return {
        "selected": selected_path,
        "all_valid": all_valid_path,
    }


# ============================================================
# 3) MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    # data/model
    ap.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    ap.add_argument("--csv", type=str, default=DEFAULT_REF_CSV)
    ap.add_argument("--ingredients_csv", type=str, default=DEFAULT_INGREDIENTS_CSV)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    # sampling
    ap.add_argument("--n_samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dirichlet_alpha", type=float, default=1.0)
    ap.add_argument("--sparse_alpha", action="store_true")

    # time window
    ap.add_argument("--start", type=float, default=168)
    ap.add_argument("--end", type=float, default=195)

    # formulation settings
    ap.add_argument("--sbm_g", type=float, default=SBM_G_PER_1000)
    ap.add_argument("--fixed_g", type=float, default=FIXED_ADDITIVES_G)
    ap.add_argument("--oil_sum_min_g", type=float, default=OIL_SUM_MIN_G)
    ap.add_argument("--oil_sum_max_g", type=float, default=OIL_SUM_MAX_G)

    # mode
    ap.add_argument(
        "--mode",
        type=str,
        default="free",
        choices=["free", "fixed_cf"],
        help="free = original broad search; fixed_cf = constrain crude fat tightly"
    )

    # constraints
    ap.add_argument("--q_low", type=float, default=0.05)
    ap.add_argument("--q_high", type=float, default=0.95)
    ap.add_argument("--sfa_limit", type=float, default=0.0060)
    ap.add_argument("--mufa_limit", type=float, default=0.0100)

    ap.add_argument(
        "--pred_dha_cap",
        type=float,
        default=150.0,
        help="Upper limit for predicted DHA when selecting the final two diets"
    )

    ap.add_argument(
        "--near_cap_quantile",
        type=float,
        default=0.90,
        help="Among diets below cap, keep the top fraction closest to the cap before diversity selection"
    )
    # crude fat target mode
    ap.add_argument("--crude_fat_target", type=float, default=None)
    ap.add_argument("--crude_fat_tol", type=float, default=None)

    # selection
    ap.add_argument(
        "--high_score_quantile",
        type=float,
        default=0.95,
        help="Pool for second diet selection: candidates with pred_DHA >= this quantile"
    )

    # optional explicit oil list
    ap.add_argument(
        "--oil_names",
        type=str,
        default="",
        help="Comma-separated oil names to keep. Empty means use all non-corn/non-SBM ingredients."
    )

    args = ap.parse_args()
    safe_mkdir(args.out_dir)

    # --------------------------------------------------------
    # A) model + reference
    # --------------------------------------------------------
    model, features, is_log, df_raw, df_ref, base_pred = prepare_model_base(
        pkl_path=args.pkl,
        ref_csv=args.csv,
        start=args.start,
        end=args.end,
    )

    # --------------------------------------------------------
    # B) ingredient library
    # --------------------------------------------------------
    oil_names_keep = [x.strip() for x in args.oil_names.split(",") if x.strip()] or None

    corn, sbm, oils_df, fa_cols_ing, oil_names = load_ingredient_library(
        ingredients_csv=args.ingredients_csv,
        oil_names_keep=oil_names_keep,
    )

    corn_fa = corn[fa_cols_ing].astype(float).values
    sbm_fa = sbm[fa_cols_ing].astype(float).values
    a_ing = oils_df[fa_cols_ing].astype(float).values.T  # (10 FA, n_oils)

    # --------------------------------------------------------
    # C) sample oil formulations directly
    # --------------------------------------------------------
    x_oil_g = dirichlet_oil_sampling(
        n_samples=args.n_samples,
        n_oils=len(oil_names),
        oil_sum_min_g=args.oil_sum_min_g,
        oil_sum_max_g=args.oil_sum_max_g,
        seed=args.seed,
        alpha=args.dirichlet_alpha,
        sparse_alpha=args.sparse_alpha,
    )

    fa_df, oil_df = build_fa_from_oils(
        x_oil_g=x_oil_g,
        oil_names=oil_names,
        a_ing=a_ing,
        corn_fa=corn_fa,
        sbm_fa=sbm_fa,
        sbm_g_per_1000=args.sbm_g,
        fixed_additives_g=args.fixed_g,
    )

    # --------------------------------------------------------
    # D) build model feature table and apply constraints
    # --------------------------------------------------------
    cand = build_candidate_feature_df(
        fa_df=fa_df,
        features=features,
        df_ref=df_ref,
        start=args.start,
        end=args.end,
    )

    cand_full = pd.concat([cand.reset_index(drop=True), oil_df.reset_index(drop=True)], axis=1)

    cand_full = apply_constraints(
        cand=cand_full,
        df_ref=df_ref,
        mode=args.mode,
        q_low=args.q_low,
        q_high=args.q_high,
        sfa_limit=args.sfa_limit,
        mufa_limit=args.mufa_limit,
        crude_fat_target=args.crude_fat_target,
        crude_fat_tol=args.crude_fat_tol,
    )

    if len(cand_full) == 0:
        raise RuntimeError(
            "No valid formulations remain after constraints.\n"
            "Try:\n"
            "- lowering sfa_limit / mufa_limit\n"
            "- widening q_low/q_high if mode=free\n"
            "- widening crude_fat_tol if mode=fixed_cf\n"
            "- increasing oil_sum_max_g or changing candidate oil set"
        )

    # --------------------------------------------------------
    # E) predict
    # --------------------------------------------------------
    raw_y = np.asarray(model.predict(cand_full[features]), dtype=float)
    yhat = np.expm1(raw_y) if is_log else raw_y

    cand_full["pred_DHA"] = yhat
    cand_full["improve_pct"] = (cand_full["pred_DHA"] - base_pred) / (base_pred + 1e-12) * 100.0

    # sort full valid list for export
    cand_full = cand_full.sort_values("pred_DHA", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------
    # F) select exactly two diets
    # --------------------------------------------------------
    selected_df = select_two_diverse_below_cap(
        cand_full=cand_full,
        pred_dha_cap=args.pred_dha_cap,
        near_cap_quantile=args.near_cap_quantile,
    )

    # --------------------------------------------------------
    # G) save
    # --------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = save_outputs(
        out_dir=args.out_dir,
        stamp=stamp,
        selected_df=selected_df,
        all_valid=cand_full,
    )

    # --------------------------------------------------------
    # H) report
    # --------------------------------------------------------
    print("\n================ Ingredient-space formulation screening =================")
    print(f"Mode: {args.mode}")
    print(f"Base Prediction: {base_pred:.6f}")
    print(f"Model features used: {len(features)}")
    print(f"Candidate oils ({len(oil_names)}): {oil_names}")
    print(f"Raw sampled formulations: {args.n_samples}")
    print(f"Valid formulations after constraints: {len(cand_full)}")
    print(f"Selected diets: {len(selected_df)}")

    if args.mode == "fixed_cf":
        print(f"Crude fat target: {args.crude_fat_target}")
        print(f"Crude fat tolerance: ±{args.crude_fat_tol}")
    else:
        print(f"Reference TotalFat quantile range: [{args.q_low}, {args.q_high}]")

    show_cols = [
        "SelectedGroup",
        "selection_note",
        "pred_DHA",
        "improve_pct",
        "Oil_g_per_kg_total",
        "Corn_g_per_kg",
        "SBM_g_per_kg",
        "FixedAdditives_g_per_kg",
        "SFA_derived,g/g diet",
        "MUFA_derived,g/g diet",
        "PUFA_derived,g/g diet",
        "n-3_derived,g/g diet",
        "n-6_derived,g/g diet",
        "TotalFat_derived,g/g diet",
    ] + [c for c in IND_FA_COLS_MODEL if c in selected_df.columns] + [
        c for c in selected_df.columns if c.endswith("_g_per_kg") and c not in {
            "Oil_g_per_kg_total", "Corn_g_per_kg", "SBM_g_per_kg", "FixedAdditives_g_per_kg"
        }
    ]

    existing_show_cols = [c for c in show_cols if c in selected_df.columns]

    print("\nSelected 2 diets for validation:")
    print(selected_df[existing_show_cols].to_string(index=False))

    print("\nSaved files:")
    print(paths["selected"])
    print(paths["all_valid"])
    print("========================================================================\n")


if __name__ == "__main__":
    main()