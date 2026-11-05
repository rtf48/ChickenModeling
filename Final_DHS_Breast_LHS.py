"""
Final_DHS_Breast_LHS.py  (IND FA version)

功能：
- 基于锁定模型进行大规模 LHS 采样
- ✅ 适配“个体脂肪酸谱”输入（10个FA列）
- ✅ 自动派生 SFA/MUFA/n3/n6/PUFA/TotalFat（用于约束与汇报）
- ✅ 物理下限：SFA_derived > 0.006 & MUFA_derived > 0.010
- ✅ 总脂肪范围：TotalFat_derived 在参考数据 5-95% 分位数内
- top4：greedy diverse selection
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


# -----------------------------
# 基础工具函数
# -----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def lhs_sample(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n + 1)
    a = cut[:n]
    b = cut[1 : n + 1]
    H = np.zeros((n, d), dtype=float)
    for j in range(d):
        u_j = rng.random(n)
        pts = a + u_j * (b - a)
        rng.shuffle(pts)
        H[:, j] = pts
    return H


def scale_to_bounds(U: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    X = np.empty_like(U, dtype=float)
    for j, (lo, hi) in enumerate(bounds):
        X[:, j] = lo + U[:, j] * (hi - lo)
    return X


def greedy_diverse_topk(
    X: np.ndarray,
    score: np.ndarray,
    k: int = 4,
    pool_size: int = 1000,
    seed: int = 42,
) -> List[int]:
    idx_sorted = np.argsort(-score)
    pool = idx_sorted[: min(pool_size, len(idx_sorted))]
    selected = [int(pool[0])]
    if k == 1:
        return selected

    X_pool = X[pool]
    std = X_pool.std(axis=0)
    std[std == 0] = 1.0
    Xn = X_pool / std

    selected_pool_pos = [0]
    for _ in range(1, k):
        sel = Xn[selected_pool_pos]
        diffs = Xn[:, None, :] - sel[None, :, :]
        dists = np.sqrt((diffs**2).sum(axis=2))
        min_dist = dists.min(axis=1)

        score_pool = score[pool]
        sp = (score_pool - score_pool.min()) / (score_pool.max() - score_pool.min() + 1e-12)

        objective = min_dist + 0.005 * sp
        objective[selected_pool_pos] = -np.inf
        best_pos = int(np.argmax(objective))
        selected_pool_pos.append(best_pos)
        selected.append(int(pool[best_pos]))
    return selected


# -----------------------------
# 个体脂肪酸列（你给的）
# -----------------------------
IND_FA_COLS = [
    "C14,g",
    "C16:0,g",
    "C16:1,g",
    "C18:0,g",
    "C18:1,g",
    "C18:2 cis n-6 LA,g",
    "C18:3 cis n-3 ALA,g",
    "C20:4n-6 ARA,g",
    "C20:5n-3 EPA,g",
    "C22:6n-3 DHA,g"
]


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    由10个个体FA派生：
    SFA = C14 + C16:0 + C18:0
    MUFA = C16:1 + C18:1
    n-3 = ALA + EPA + DHA
    n-6 = LA + ARA
    PUFA = n-3 + n-6
    TotalFat = sum(10 FAs)
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


# -----------------------------
# 约束逻辑（个体FA版本）
# -----------------------------
def apply_constraints(
    cand: pd.DataFrame,
    df_ref: pd.DataFrame,
    q_low: float,
    q_high: float,
    eps: float = 1e-12,
    sfa_limit: float = 0.0060,
    mufa_limit: float = 0.0100,
) -> pd.DataFrame:
    c = cand.copy()

    # 时间窗长度
    if all(k in c.columns for k in ["Start", "End", "Window Length"]):
        c["Window Length"] = c["End"] - c["Start"] + 1

    # 派生指标
    c = add_derived_metrics(c)

    # 1) 物理下限
    c = c.loc[c["SFA_derived,g/g diet"] >= sfa_limit].copy()
    c = c.loc[c["MUFA_derived,g/g diet"] >= mufa_limit].copy()

    if len(c) == 0:
        return c.reset_index(drop=True)

    # 2) 总脂肪范围（基于参考数据分位数）
    ref = add_derived_metrics(df_ref)
    ref_total = pd.to_numeric(ref["TotalFat_derived,g/g diet"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not ref_total.empty:
        lo, hi = float(ref_total.quantile(q_low)), float(ref_total.quantile(q_high))
        c = c.loc[
            (c["TotalFat_derived,g/g diet"] >= lo - eps)
            & (c["TotalFat_derived,g/g diet"] <= hi + eps)
        ].copy()

    return c.reset_index(drop=True)


# -----------------------------
# 执行逻辑
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default="outputs/IND_Breast_DHA.pkl")
    ap.add_argument("--csv", type=str, default="Broiler_Calculated.csv")
    ap.add_argument("--n_samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/phase_specific")
    ap.add_argument("--start", type=float, default=168)
    ap.add_argument("--end", type=float, default=195)

    # 分位数约束
    ap.add_argument("--q_low", type=float, default=0.05)
    ap.add_argument("--q_high", type=float, default=0.95)

    # 物理下限
    ap.add_argument("--sfa_limit", type=float, default=0.0060)
    ap.add_argument("--mufa_limit", type=float, default=0.0100)

    args = ap.parse_args()
    safe_mkdir(args.out_dir)

    bundle = joblib.load(args.pkl)
    model, features = bundle["model"], list(bundle["features"])
    is_log = bundle.get("log_transformed", False)

    df_raw = pd.read_csv(args.csv)

    # 检查 CSV 是否包含所有 features
    missing = [c for c in features if c not in df_raw.columns]
    if missing:
        raise KeyError(
            "你的 CSV 缺少锁定模型所需的特征列：\n"
            + "\n".join(missing)
            + "\n\n请确认 Broiler_Calculated.csv 的列名与 pkl 中 features 完全一致。"
        )

    df_ref = coerce_numeric(df_raw[features], features)

    # -----------------------------
    # 1) Base 预测
    # -----------------------------
    base_vec = df_ref.median(numeric_only=True)
    base = pd.DataFrame([base_vec[features].values], columns=features)

    if "Start" in base.columns:
        base["Start"] = args.start
    if "End" in base.columns:
        base["End"] = args.end
    if "Window Length" in base.columns:
        base["Window Length"] = args.end - args.start + 1

    raw_base_pred = float(model.predict(base[features])[0])
    base_pred = np.expm1(raw_base_pred) if is_log else raw_base_pred

    # -----------------------------
    # 2) LHS 采样
    # -----------------------------
    time_cols = [c for c in ["Start", "End", "Window Length"] if c in features]
    sample_cols = [c for c in features if c not in time_cols]

    bounds: List[Tuple[float, float]] = []
    for c in sample_cols:
        lo = float(df_ref[c].quantile(args.q_low))
        hi = float(df_ref[c].quantile(args.q_high))
        bounds.append((lo, hi))

    U = lhs_sample(args.n_samples, len(sample_cols), args.seed)
    X_samp = scale_to_bounds(U, bounds)
    cand = pd.DataFrame(X_samp, columns=sample_cols)

    if "Start" in features:
        cand["Start"] = args.start
    if "End" in features:
        cand["End"] = args.end
    if "Window Length" in features:
        cand["Window Length"] = args.end - args.start + 1

    # -----------------------------
    # 3) 约束过滤
    # -----------------------------
    cand = apply_constraints(
        cand=cand,
        df_ref=df_ref,
        q_low=args.q_low,
        q_high=args.q_high,
        sfa_limit=args.sfa_limit,
        mufa_limit=args.mufa_limit,
    )

    if len(cand) == 0:
        raise RuntimeError(
            "约束过滤后候选集为空。\n"
            "尝试：降低 sfa_limit/mufa_limit 或放宽 q_low/q_high。"
        )

    # -----------------------------
    # 4) 预测与筛选
    # -----------------------------
    raw_y = np.asarray(model.predict(cand[features]), dtype=float)
    yhat = np.expm1(raw_y) if is_log else raw_y

    # 多样性空间：优先用 diet 相关列（g/g diet），否则退回 sample_cols
    div_cols = [c for c in features if "g/g diet" in c]
    if not div_cols:
        div_cols = sample_cols

    X_div = cand[div_cols].to_numpy()
    idx_sel = greedy_diverse_topk(X_div, yhat, k=4, seed=args.seed)

    top4 = cand.iloc[idx_sel].copy()
    top4["pred_DHA"] = yhat[idx_sel]
    top4["improve_pct"] = (top4["pred_DHA"] - base_pred) / (base_pred + 1e-12) * 100

    # 派生列用于打印/保存（可删除也行）
    top4 = add_derived_metrics(top4)

    # -----------------------------
    # 5) 输出
    # -----------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"diverse_top4_DHA_INDFA_{stamp}.csv")
    top4.to_csv(out_path, index=False)

    print(f"\n================ 强多样性模拟报告 (个体FA输入 + 双重物理限制版) =================")
    print(f"Base Prediction: {base_pred:.6f}")
    print(f"Valid Samples (after constraints): {len(cand)}")

    show_cols = [
        "pred_DHA",
        "improve_pct",
        "SFA_derived,g/g diet",
        "MUFA_derived,g/g diet",
        "PUFA_derived,g/g diet",
        "n-3_derived,g/g diet",
        "n-6_derived,g/g diet",
        "TotalFat_derived,g/g diet",
    ]
    show_cols += [c for c in IND_FA_COLS if c in top4.columns]
    print(top4[show_cols].to_string(index=False))

    print(f"\nSaved: {out_path}")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()