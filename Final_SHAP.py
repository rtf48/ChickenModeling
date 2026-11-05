# Final_SHAP_Updated.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import shap

# 导入你的数据集模块和特征列表模块
import Final_listStorage_Layer as ls
import Final_dataset as dataset

# ---------------------------------------------------------
# 1. 锁定模型的最优超参数 (与 Final_lock_model 脚本完全一致)
# ---------------------------------------------------------
BEST_PARAMS_GB = {
    'subsample': 0.7,
    'n_estimators': 2000,
    'max_depth': 5,
    'learning_rate': 0.02,
    'min_samples_split': 10,
    'max_features': 0.7,
    'random_state': 42
}

OUT_ROOT = 'outputs/shap'
TOPK_MAIN = 6  # 绘制前 6 个重要特征的主效应图
TOPM_INTER = 6  # 绘制前 6 对强交互作用图
DPI = 160  # 图片清晰度


# ----------------------------
# 2. 工具函数
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def sanitize(s):
    """清理文件名中的非法字符"""
    return s.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')


def fit_full_gb(features, target):
    """
    获取数据，执行缺失值填充，并进行 Log 变换后再训练
    """
    X, y = dataset.get_data(features, target)
    X = X.fillna(X.mean(numeric_only=True))

    # 【核心逻辑】执行 Log 变换，确保解释的是优化后的模型行为
    y_log = np.log1p(y)

    model = GradientBoostingRegressor(**BEST_PARAMS_GB)
    model.fit(X, y_log)
    return model, X, y_log


def rank_topk(values: np.ndarray, feat_names, k: int):
    """根据平均绝对 SHAP 值对特征进行排序"""
    mean_abs = np.mean(np.abs(values), axis=0)
    idx = np.argsort(-mean_abs)[:k]
    return [feat_names[i] for i in idx], mean_abs[idx].tolist(), mean_abs


def top_inter_pairs(inter_vals: np.ndarray, feat_names, m: int):
    """找出交互作用最强的特征对"""
    mean_abs = np.mean(np.abs(inter_vals), axis=0)  # 形状为 [n_feat, n_feat]
    n = mean_abs.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # 交互矩阵是对称的，取其中一半即可
            pairs.append((i, j, mean_abs[i, j]))
    pairs.sort(key=lambda t: -t[2])
    pairs = pairs[:m]
    return [(feat_names[i], feat_names[j], float(s)) for i, j, s in pairs]


# ----------------------------
# 3. 核心运行逻辑
# ----------------------------
def run_one_target(features, target):
    print(f"[SHAP] Starting analysis for: {target}")
    outdir = ensure_dir(os.path.join(OUT_ROOT, f"{target}__gb"))

    # 训练模型
    model, X, y_log = fit_full_gb(features, target)
    feat_names = list(X.columns)

    # 创建 SHAP 解释器 (针对树模型使用 TreeExplainer)
    explainer = shap.TreeExplainer(model)
    shap_exp = explainer(X)
    values = shap_exp.values  # shape: [n_samples, n_features]

    # A. Summary Plot (蜂窝图：展示特征对预测的正负向驱动)
    plt.figure()
    shap.summary_plot(shap_exp, X, show=False)
    plt.title(f"SHAP Summary: {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{target}__gb__summary.png"), dpi=DPI)
    plt.close()

    # B. Feature Importance Bar Plot (柱状图：展示特征的平均贡献度)
    plt.figure()
    shap.summary_plot(shap_exp, X, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{target}__gb__bar.png"), dpi=DPI)
    plt.close()

    # C. Dependence Plots (主效应图：展示单个特征值与预测贡献的非线性关系)
    top_feats, _, _ = rank_topk(values, feat_names, TOPK_MAIN)
    for f in top_feats:
        try:
            plt.figure()
            # 自动寻找最具交互性的特征作为颜色区分
            shap.dependence_plot(f, values, X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{target}__gb__depend_{sanitize(f)}.png"), dpi=DPI)
            plt.close()
        except Exception as e:
            print(f"[WARN] Dependence plot failed for {f}: {e}")

    # D. 导出 CSV 数据表 (方便之后用 Origin/R 自定义绘图)
    shap_df = pd.DataFrame(values, columns=feat_names)
    y_pred_log = model.predict(X)
    out_df = pd.concat([
        pd.Series(y_log.values, name='target_true_log'),
        pd.Series(y_pred_log, name='target_pred_log'),
        shap_df,
        X.reset_index(drop=True)
    ], axis=1)
    out_df.to_csv(os.path.join(outdir, f"{target}__gb__shap_table.csv"), index=False)

    # E. Interaction Analysis (交互作用分析：极其耗时，但对营养学研究很有意义)
    try:
        print(f"[SHAP] Calculating interaction values for {target}...")
        inter_vals = shap.TreeExplainer(model).shap_interaction_values(X)
        np.save(os.path.join(outdir, f"{target}__gb__shap_interactions.npy"), inter_vals)

        # 交互作用 Summary Plot
        plt.figure()
        shap.summary_plot(inter_vals, X, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{target}__gb__summary_interactions.png"), dpi=DPI)
        plt.close()

        # 针对前 M 对强交互特征绘图
        for fa, fb, _score in top_inter_pairs(inter_vals, feat_names, TOPM_INTER):
            try:
                plt.figure()
                shap.dependence_plot(fa, values, X, interaction_index=fb, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"{target}__gb__depend_{sanitize(fa)}_x_{sanitize(fb)}.png"), dpi=DPI)
                plt.close()
            except Exception as e:
                print(f"[WARN] Interaction plot failed {fa} x {fb}: {e}")
    except Exception as e:
        print(f"[WARN] Interaction analysis skipped for {target}: {e}")

    # F. 保存元数据报告
    meta = {
        "best_params": BEST_PARAMS_GB,
        "target": target,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "top_features_main": top_feats,
        "log_transformed": True
    }
    with open(os.path.join(outdir, f"{target}__gb__meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[SHAP] Successfully completed: {target}")


# ----------------------------
# 4. 主入口
# ----------------------------
def main():
    features = ls.target_features_comp  # 使用你在 storage 文件中定义的特征列表

    # 如果你想跑所有 5 个核心指标，使用 ls.target_labels_5
    # 如果只想跑 Breast DHA，可以改为 targets = ["Breast C22:6"]
    targets = ls.target_labels_5

    ensure_dir(OUT_ROOT)
    for t in targets:
        try:
            run_one_target(features, t)
        except Exception as e:
            print(f"[ERROR] Critical failure for {t}: {e}")

    print("\n" + "=" * 50)
    print("SHAP Analysis Finished with Optimized Parameters.")
    print("Check 'outputs/shap' for all plots and tables.")
    print("=" * 50)


if __name__ == '__main__':
    main()