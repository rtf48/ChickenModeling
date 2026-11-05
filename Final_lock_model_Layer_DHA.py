from __future__ import annotations

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score


# ----------------------------
# 工具函数 (确保包含在脚本内)
# ----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df = df.copy()
    df.columns = new_cols
    return df


def load_features_from_liststorage(manual_features: Optional[List[str]] = None) -> List[str]:
    features = None
    try:
        import Final_listStorage_Layer as ls
        if hasattr(ls, "target_features_comp"):
            features = list(ls.target_features_comp)
    except Exception:
        features = None
    if manual_features:
        features = manual_features
    if features is None:
        features = ["Start", "End", "Window Length", 'C14,g/g diet',
    'C16:0,g/g diet',
    'C16:1,g/g diet',
    'C18:0,g/g diet',
    'C18:1,g/g diet',
    'C18:2 cis n-6 LA,g/g diet',
    'C18:3 cis n-3 ALA,g/g diet']
    return [c for c in features if "..." not in c]


def resolve_feature_names(features: List[str], df_cols: List[str]) -> List[str]:
    resolved = []
    for f in features:
        if f in df_cols:
            resolved.append(f)
            continue
        dup_candidates = [c for c in df_cols if c.startswith(f + "__dup")]
        resolved.append(dup_candidates[0] if dup_candidates else f)
    return resolved


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("model", GradientBoostingRegressor(random_state=seed)),
    ])


# ----------------------------
# 核心逻辑 (修复了引用错误)
# ----------------------------
def oof_cv_with_search(X: pd.DataFrame, y: pd.Series, seed: int, cv_folds: int, n_iter: int, n_jobs: int,
                       verbose: int) -> Tuple[np.ndarray, List[float], Dict]:
    # 训练前进行 Log 变换
    y_log = np.log1p(y)

    param_dist = {
        "model__n_estimators": [1000, 1500, 2000],  # 大胆增加树的数量，让模型有更多机会纠偏
        "model__learning_rate": [0.01, 0.02, 0.05],  # 降低学习率，配合更多的树，实现“慢工出细活”
        "model__max_depth": [3, 4, 5],  # 【重点】加深树深，允许模型学习更复杂的交互逻辑
        "model__subsample": [0.6, 0.7, 0.8],  # 稍微降低子采样率，通过增加随机性来压制深树带来的过拟合
        "model__min_samples_split": [2, 5, 10],  # 增加分裂节点的限制，平衡深树
        "model__max_features": [0.7, 0.8, 0.9]
    }

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_pred_raw = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)
    fold_r2 = []
    best_fold_r2 = -1e9
    best_params_selected = {}

    for tr_idx, va_idx in kf.split(X):
        X_tr, y_tr_log = X.iloc[tr_idx], y_log.iloc[tr_idx]
        X_va, y_va_raw = X.iloc[va_idx], y.iloc[va_idx]

        search = RandomizedSearchCV(
            estimator=build_pipeline(seed),
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",
            cv=3, random_state=seed, n_jobs=n_jobs
        )
        search.fit(X_tr, y_tr_log)

        pred_log = search.best_estimator_.predict(X_va)
        pred_va_raw = np.expm1(pred_log)  # 还原到原始尺度评估

        oof_pred_raw[va_idx] = pred_va_raw
        r2 = r2_score(y_va_raw, pred_va_raw)
        fold_r2.append(float(r2))

        if r2 > best_fold_r2:
            best_fold_r2 = r2
            best_params_selected = dict(search.best_params_)

    return oof_pred_raw, fold_r2, best_params_selected


from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default='/Users/robinli/Downloads/mps project 2/Database Master/Dataset/Project2_Layers_Individual_FA_corrected_without49.csv'
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Egg C22:6"
    )
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--search_iter", type=int, default=30)
    args = parser.parse_args()

    safe_mkdir(args.out_dir)
    features = load_features_from_liststorage()
    df = pd.read_csv(args.csv)
    df = normalize_duplicate_columns(df)
    features_resolved = resolve_feature_names(features, df.columns.tolist())

    # 准备数据并处理 NaN
    mask = df[args.target].notna()
    X = df.loc[mask, features_resolved].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    y = df.loc[mask, args.target].reset_index(drop=True)

    # 评估与锁定 (OOF 预测是在原始尺度还原后的数值)
    oof_pred, fold_r2, best_params = oof_cv_with_search(X, y, args.seed, args.cv_folds, args.search_iter, -1, 1)

    # --- 计算额外的模型表现指标 ---
    overall_r2 = r2_score(y, oof_pred)
    overall_rmse = root_mean_squared_error(y, oof_pred)
    overall_mae = mean_absolute_error(y, oof_pred)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import matplotlib.pyplot as plt

    # ===== Predicted vs Actual scatter plot =====
    plt.figure(figsize=(6, 6))

    plt.scatter(
        y,  # actual
        oof_pred,  # predicted
        alpha=0.7,
        edgecolor='k'
    )

    # y=x reference line
    min_val = min(y.min(), oof_pred.min())
    max_val = max(y.max(), oof_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2,
        label="Perfect prediction"
    )

    plt.xlabel("Actual Breast DHA")
    plt.ylabel("Predicted Breast DHA")
    plt.title("Predicted vs Actual Breast DHA (OOF)")
    plt.legend()
    plt.grid(True)

    # 保存
    scatter_path = os.path.join(args.out_dir, f"predicted_vs_actual_{stamp}.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    print(f"Scatter plot saved to: {scatter_path}")

    # 打印详细表现
    print("\n================= 模型表现报告 (OOF) =================")
    print(f"目标变量: {args.target}")
    print(f"样本总量: {len(y)}")
    print(f"平均 R2 (折间平均): {np.mean(fold_r2):.4f}")
    print(f"整体 R2: {overall_r2:.4f}")
    print(f"整体 RMSE: {overall_rmse:.4f}")
    print(f"整体 MAE: {overall_mae:.4f}")
    print("====================================================\n")

    # 锁定模型逻辑保持不变
    final_pipe = build_pipeline(args.seed)
    if best_params:
        final_pipe.set_params(**best_params)
    final_pipe.fit(X, np.log1p(y))

    # 保存
    pkl_path = os.path.join(args.out_dir, f"locked_model_log_{stamp}.pkl")
    perf_path = os.path.join(args.out_dir, f"performance_log_{stamp}.json")

    # 扩展 bundle 信息
    bundle = {
        "model": final_pipe,
        "features": features_resolved,
        "target": args.target,
        "log_transformed": True,
        "oof_r2": overall_r2,
        "fold_r2": fold_r2,
        "rmse": overall_rmse,
        "mae": overall_mae
    }
    joblib.dump(bundle, pkl_path)

    # 同时保存一份 JSON 报告，方便直接查阅数值
    with open(perf_path, 'w') as f:
        json.dump({
            "target": args.target,
            "metrics": {
                "r2": overall_r2,
                "rmse": overall_rmse,
                "mae": overall_mae,
                "fold_r2": fold_r2
            },
            "best_params": best_params
        }, f, indent=4)

    print(f"Success! 模型已保存至: {pkl_path}")
    print(f"性能报告已保存至: {perf_path}")


if __name__ == "__main__":
    main()