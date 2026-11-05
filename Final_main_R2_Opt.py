import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import Final_dataset as dataset   # 按你实际文件名来

# 你自己的模块（保持不动）
import Final_listStorage_Broiler as ls

# ============== 原有简易训练器（可能被其他地方调用，保留即可） ==============
def train_rf(data_train, labels_train, n_estimators=100, max_depth=5):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(data_train, labels_train)
    return model

def train_lr(data_train, labels_train):
    model = LinearRegression()
    model.fit(data_train, labels_train)
    return model

def train_nn(data_train, labels_train, hidden_layer_sizes=(64, 32),
             activation='relu', solver='adam', max_iter=2000):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation, solver=solver, max_iter=max_iter)
    model.fit(data_train, labels_train)
    return model

def train_gb(data_train, labels_train, n_estimators=1000, learning_rate=0.1,
             max_depth=5, subsample=0.8, validation_fraction=0.2,
             n_iter_no_change=10, tol=0.0001):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=subsample, validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change, tol=tol
    )
    model.fit(data_train, labels_train)
    return model

def train_svm(data_train, labels_train, kernel='rbf', C=1.0, epsilon=0.1):
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(data_train, labels_train)
    return model

# ============== 实用函数：MAPE 防 0（不为 outlier，主要防分母为 0） ==============
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))

# ============== 日志容器：记录每折最优参数 ==============
best_params_log = {}

# ============== 模型与搜索空间（只动当前脚本，不改 dataset / listStorage） ==============
def build_model_and_searchspace(name, random_state=42):
    if name == 'rf':
        est = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        params = {
            'model__n_estimators': [200, 400, 800, 1200],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 0.7, 1.0]
        }
        return est, params

    if name == 'gb':
        est = GradientBoostingRegressor(random_state=random_state)
        params = {
            'model__n_estimators': [300, 600, 1000],
            'model__learning_rate': [0.03, 0.05, 0.1, 0.2],
            'model__max_depth': [2, 3, 5],
            'model__subsample': [0.6, 0.8, 1.0]
        }
        return est, params

    if name == 'svm':
        est = SVR(kernel='rbf')
        params = {
            'model__C': [0.1, 1, 10, 50, 100],
            'model__gamma': ['scale', 0.01, 0.03, 0.1]
        }
        return est, params

    if name == 'nn':
        est = MLPRegressor(random_state=random_state, max_iter=3000)
        params = {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [1e-5, 1e-4, 1e-3],
            'model__learning_rate_init': [1e-4, 3e-4, 1e-3]
        }
        return est, params

    if name == 'lr':
        est = LinearRegression()
        params = {}  # 线性回归基本无关键超参
        return est, params

    raise ValueError(f'Unknown model name: {name}')


# ============== 折内：随机搜索调参，返回本折最优模型并记录最优参数 ==============
def tune_in_fold(model_name, X_train, y_train, random_state=42, n_iter=25):
    base_est, params = build_model_and_searchspace(model_name, random_state=random_state)

    # 统一前处理：折内拟合 → 避免泄漏
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("model", base_est)
    ])

    if len(params) == 0:
        pipe.fit(X_train, y_train)
        best_params_log.setdefault(model_name, []).append({'_no_params_': True})
        return pipe

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=n_iter,
        scoring='r2',
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_params_log.setdefault(model_name, []).append(search.best_params_)
    return search.best_estimator_


# ============== 外层：KFold，每折先调参再预测，最后汇总指标 ==============
def compare_models(features, target, random_state=42, n_splits=5, n_iter=25):
    # 不改 dataset 接口
    data, labels = dataset.get_data(features, target)

    model_names = ['rf', 'lr', 'svm', 'nn', 'gb']
    metrics = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name in model_names:
        preds = np.zeros_like(labels, dtype=float)

        for train_index, test_index in kf.split(data):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            best_model = tune_in_fold(name, X_train, y_train,
                                      random_state=random_state, n_iter=n_iter)
            preds[test_index] = best_model.predict(X_test)

        rmse = root_mean_squared_error(labels, preds)
        r2 = r2_score(labels, preds)
        mape = safe_mape(labels, preds)  # 防 0
        metrics[name] = {'rmse': rmse, 'r2': r2, 'mape': mape}
        print(f"{name} | RMSE: {rmse:.3f}, R2: {r2:.3f}, MAPE: {mape:.3f}")

    return metrics

# ============== 批量评估多个 target，结构与原来一致（修复列名 rnn→nn） ==============
def eval_all(features, targets, random_state=42, n_splits=5, n_iter=25):
    results = pd.DataFrame(columns=[
        'rf rmse', 'rf r2', 'rf mape',
        'lr rmse', 'lr r2', 'lr mape',
        'svm rmse', 'svm r2', 'svm mape',
        'nn rmse', 'nn r2', 'nn mape',
        'gb rmse', 'gb r2', 'gb mape'
    ])

    failed_runs = []
    for t in targets:
        try:
            metrics = compare_models(features, t,
                                     random_state=random_state,
                                     n_splits=n_splits, n_iter=n_iter)
            results.loc[t] = [
                metrics['rf']['rmse'], metrics['rf']['r2'], metrics['rf']['mape'],
                metrics['lr']['rmse'], metrics['lr']['r2'], metrics['lr']['mape'],
                metrics['svm']['rmse'], metrics['svm']['r2'], metrics['svm']['mape'],
                metrics['nn']['rmse'], metrics['nn']['r2'], metrics['nn']['mape'],
                metrics['gb']['rmse'], metrics['gb']['r2'], metrics['gb']['mape']
            ]
        except Exception as e:
            print(f"[WARN] Training failed for {t}: {e}")
            failed_runs.append(t)

    if len(failed_runs) > 0:
        print(f"Training failed for {failed_runs}")
    return results

# ============== 保存结果 ==============
def save_output_csv(name, df):
    with open(f'outputs/{name}.csv', "w") as file:
        file.write(df.to_csv())

def save_best_params_csv(name='best-params-log'):
    """
    把每个模型、每一折的 best_params_ 记录到 CSV。
    列：model, fold, best_params_json
    """
    rows = []
    for model_name, params_list in best_params_log.items():
        for idx, params in enumerate(params_list, start=1):
            rows.append({
                'model': model_name,
                'fold': idx,
                'best_params_json': json.dumps(params)
            })
    df = pd.DataFrame(rows)
    df.to_csv(f'outputs/{name}.csv', index=False)
    print(f"[INFO] Best params saved to outputs/{name}.csv")

# ============== 入口调用（保持你的用法；特征和标签都用 ls 提供的） ==============
# 提醒：把你想用的 9 个输入变量放进 ls.target_features_comp 即可，或在下一行把第1参改成手动列表。
compare_all = eval_all(ls.target_features_comp, ls.target_labels_5, n_iter=25, n_splits=5)
save_output_csv('full-feature-comparison', compare_all)
save_best_params_csv('best-params-log')
