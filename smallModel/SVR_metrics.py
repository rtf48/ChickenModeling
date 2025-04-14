import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error

# Load
dataset = pd.read_csv("Dataset_3_12.csv")
dataset["Start"] = pd.to_numeric(dataset["Start"], errors="coerce")
dataset["End"] = pd.to_numeric(dataset["End"], errors="coerce")
dataset.columns = dataset.columns.str.strip()

# features
features = ["C14,g", "C15:0,g", "C15:1,g", "C16:0,g", "C16:1,g", "C17:0,g", "C17:1,g",
            "C18:0,g", "C18:1,g", "C18:2 cis n-6 LA,g", "C18:3 cis n-3 ALA,g",
            "C20:4n-6 ARA,g", "C20:5n-3 EPA,g", "C22:6n-3 DHA,g", "C24:0,g",
            "Start", "End"]

# Outputs
targets = [
    'average feed intake g per d.1', 'bodyweightgain,g', 'akp U per ml', 'alt (U per L)',
    'glucose (g per L)', 'nefa,umol per L', 'pip mg per dL', 'tc mg per g',
    'tg mg per g', 'trap U per L', 'uric acid mmol per L', 'BCA',
    'breast mTOR', 'breast S6K1', 'breast 4E-BP1', 'breast MURF1',
    'breast MAFbx', 'breast AMPK', 'liver mTOR', 'liver S6K1',
    'liver 4E-BP1', 'liver MURF1', 'liver MAFbx', 'liver AMPK',
    'Plasma SFA', 'Plasma MUFA', 'Plasma PUFA', 'Plasma n-3', 'Plasma n-6', 'Plasma C16:0',
    'Plasma C16:1', 'Plasma C18:0', 'Plasma C18:1', 'Plasma C18:2', 'Plasma C18:3',
    'Plasma C20:5', 'Plasma C22:6', 'Liver SFA', 'Liver MUFA', 'Liver PUFA', 'Liver n-3',
    'Liver n-6', 'Liver C16:00', 'Liver C16:1', 'Liver C18:0', 'Liver C18:1', 'Liver C18:2',
    'Liver C18:3', 'Liver C20:5', 'Liver C22:6', 'Breast SFA', 'Breast MUFA', 'Breast PUFA',
    'Breast n-3', 'Breast n-6', 'Breast C16:0', 'Breast C16:01', 'Breast C18:0', 'Breast C18:01',
    'Breast C18:2', 'Breast C18:3', 'Breast C20:4', 'Breast C20:5', 'Breast C22:6', 'Thigh SFA',
    'Thigh MUFA', 'Thigh PUFA', 'Thigh n-3', 'Thigh n-6', 'Thigh C16:0', 'Thigh C16:01',
    'Thigh C18:0', 'Thigh C18:1', 'Thigh C18:2', 'Thigh C18:3', 'Thigh C20:4', 'Thigh C22:6'
]

# initialize
results_dict = {
    "Training RMSE": [],
    "Training R2": [],
    "Training MAPE": [],
    "Testing RMSE": [],
    "Testing R2": [],
    "Testing MAPE": []
}
valid_targets = []

# Outputs
for target in targets:
    if target not in dataset.columns:
        print(f"❌ Column missing：{target}")
        continue

    temp = dataset.dropna(subset=[target])
    if temp.empty or temp[target].nunique() <= 1:
        print(f"⚠️ Cannot train：{target}")
        continue

    X = temp[features]
    y = temp[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Kfold
    model = SVR(kernel='rbf', C=10, epsilon=0.05, gamma='scale')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_rmse, train_r2, train_mape = [], [], []
    test_rmse, test_r2, test_mape = [], [], []

    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_rmse.append(root_mean_squared_error(y_train, y_pred_train))
        train_r2.append(r2_score(y_train, y_pred_train))
        train_mape.append(mean_absolute_percentage_error(y_train, y_pred_train))

        test_rmse.append(root_mean_squared_error(y_test, y_pred_test))
        test_r2.append(r2_score(y_test, y_pred_test))
        test_mape.append(mean_absolute_percentage_error(y_test, y_pred_test))

    # add to dic
    results_dict["Training RMSE"].append(np.mean(train_rmse))
    results_dict["Training R2"].append(np.mean(train_r2))
    results_dict["Training MAPE"].append(np.mean(train_mape))
    results_dict["Testing RMSE"].append(np.mean(test_rmse))
    results_dict["Testing R2"].append(np.mean(test_r2))
    results_dict["Testing MAPE"].append(np.mean(test_mape))
    valid_targets.append(target)

# create dataframe
metrics = pd.DataFrame(results_dict, index=valid_targets).T
metrics.index.name = "Metric"

# save
metrics.to_csv("svr_kfold_metrics.csv")
print("✅ save as svr_kfold_metrics.csv")
