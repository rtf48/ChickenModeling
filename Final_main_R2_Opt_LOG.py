import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

# 导入你的数据集和列表模块
import Final_dataset as dataset
import Final_listStorage_Layer as ls


# ----------------------------
# 实用工具
# ----------------------------
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))


# 日志容器
best_params_log = {}


# ----------------------------
# 模型与搜索空间定义
# ----------------------------
def build_model_and_searchspace(name, random_state=42):
    if name == 'rf':
        est = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        params = {
            'model__n_estimators': [200, 400, 800],
            'model__max_depth': [None, 5, 10],
            'model__max_features': ['sqrt', 1.0]
        }
        return est, params
    if name == 'gb':
        est = GradientBoostingRegressor(random_state=random_state)
        params = {
            'model__n_estimators': [300, 600, 1000],
            'model__learning_rate': [0.03, 0.05, 0.1],
            'model__max_depth': [3, 4, 5],
            'model__subsample': [0.7, 0.8, 0.9],
            'model__max_features': [0.7, 0.8, 0.9, None]
        }
        return est, params
    if name == 'svm':
        est = SVR(kernel='rbf')
        params = {'model__C': [0.1, 1, 10, 100], 'model__gamma': ['scale', 0.01, 0.1]}
        return est, params
    if name == 'nn':
        est = MLPRegressor(random_state=random_state, max_iter=3000)
        params = {
            'model__hidden_layer_sizes': [(64, 32), (128, 64)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [1e-4, 1e-3]
        }
        return est, params
    if name == 'lr':
        return LinearRegression(), {}
    raise ValueError(f'Unknown model: {name}')


# ----------------------------
# 核心逻辑：Log 训练 + 原始尺度评估
# ----------------------------
def tune_in_fold(model_name, X_train, y_train, random_state=42, n_iter=25):
    # 【核心】对训练目标进行 log1p 变换
    y_train_log = np.log1p(y_train)

    base_est, params = build_model_and_searchspace(model_name, random_state=random_state)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("model", base_est)
    ])

    if len(params) == 0:
        pipe.fit(X_train, y_train_log)
        return pipe

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',  # 在对数空间评估 MSE 更稳健
        cv=3,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train_log)
    best_params_log.setdefault(model_name, []).append(search.best_params_)
    return search.best_estimator_


def compare_models(features, target, random_state=42, n_splits=5, n_iter=25):
    data, labels = dataset.get_data(features, target)
    model_names = ['rf', 'lr', 'svm', 'nn', 'gb']
    metrics = {}

    # 保险一点：强制转成数值，避免奇怪类型
    labels = pd.to_numeric(labels, errors='coerce')
    valid_mask = labels.notna()
    data = data.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].reset_index(drop=True)

    # 样本太少直接返回 NaN，避免整个 target 崩
    if len(labels) < 2:
        print(f"[WARN] Target={target} has fewer than 2 valid samples. Skipped.")
        for name in model_names:
            metrics[name] = {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}
        return metrics

    # 自动调整 folds，避免 n_splits > n_samples
    effective_splits = min(n_splits, len(labels))
    if effective_splits < 2:
        print(f"[WARN] Target={target} has too few samples for CV. Skipped.")
        for name in model_names:
            metrics[name] = {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}
        return metrics

    kf = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    for name in model_names:
        try:
            preds_original_scale = np.zeros(len(labels), dtype=float)

            for tr_idx, te_idx in kf.split(data):
                X_tr, y_tr = data.iloc[tr_idx], labels.iloc[tr_idx]
                X_te = data.iloc[te_idx]

                best_model = tune_in_fold(name, X_tr, y_tr, random_state, n_iter)

                # 预测 log 值并执行 expm1 还原
                y_pred_log = best_model.predict(X_te)

                # 防止 expm1 overflow，把极端值裁掉
                y_pred_log = np.clip(y_pred_log, -20, 20)

                preds_original_scale[te_idx] = np.expm1(y_pred_log)

            # 再保险：如果还是出现 inf，直接让这个模型记 NaN
            if not np.all(np.isfinite(preds_original_scale)):
                raise ValueError("Predictions contain inf or non-finite values after expm1.")

            r2 = r2_score(labels, preds_original_scale)
            rmse = root_mean_squared_error(labels, preds_original_scale)
            mae = mean_absolute_error(labels, preds_original_scale)

            metrics[name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
            print(f"Target: {target} | {name} | R2: {r2:.4f} | RMSE: {rmse:.4f}")

        except Exception as e:
            print(f"[WARN] Target={target} | Model={name} failed: {e}")
            metrics[name] = {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}

    return metrics


def eval_all(features, targets, random_state=42, n_splits=5, n_iter=25):
    all_results = []
    for t in targets:
        try:
            m = compare_models(features, t, random_state, n_splits, n_iter)
            row = {'Target': t}
            for name in m:
                row[f'{name}_r2'] = m[name]['r2']
                row[f'{name}_rmse'] = m[name]['rmse']
            all_results.append(row)
        except Exception as e:
            print(f"[WARN] Failed for {t}: {e}")

    return pd.DataFrame(all_results)


# ----------------------------
# 执行与保存
# ----------------------------
if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)

    # 运行所有指标的 Log 评估
    results_df = eval_all(ls.target_features_comp, ls.target_labels_5, n_iter=25)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'outputs/log_comparison_results_{stamp}.csv', index=False)

    print("\n[FINISH] 所有指标的 Log 变换评估已完成，结果保存至 outputs/")