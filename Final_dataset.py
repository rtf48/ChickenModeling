import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

raw = pd.read_csv('/Users/robinli/Downloads/mps project 2/Database Master/Dataset/Project2_Layers_Individual_FA_corrected_without49.csv')


def set_raw(file):
    global raw
    raw = pd.read_csv(file)


z_norm_outputs = ['breast mTOR', 'breast S6K1', 'breast 4E-BP1', 'breast MURF1',
                  'breast MAFbx', 'breast AMPK', 'liver mTOR', 'liver S6K1',
                  'liver 4E-BP1', 'liver MURF1', 'liver MAFbx', 'liver AMPK']


def z_norm(dataset, target):
    studies = split_by_study(dataset)

    z_norm = StandardScaler()

    for _, v in studies.items():
        v[target] = z_norm.fit_transform(v[target])

    return pd.concat(studies.values(), ignore_index=True)


def split_by_study(dataset):
    studies = set(dataset['Study'])
    sub_datasets = {}

    for study in studies:
        sub_datasets[study] = dataset[dataset['Study'] == study].copy()
    return sub_datasets


def add_poly_features(features):
    scaler = MinMaxScaler()
    poly = PolynomialFeatures(degree=2, include_bias=True)

    ptransform_data = pd.DataFrame(scaler.fit_transform(poly.fit_transform(features)),
                                   columns=poly.get_feature_names_out(features.columns))

    return ptransform_data


def get_data(in_vars, out_var):
    temp = raw.dropna(subset=[out_var]).copy()

    # 如果你需要对 y 做按 Study 的 z 标准化（注意：这一步会有跨研究的评估偏差问题）
    if out_var in z_norm_outputs:
        temp = z_norm(temp, out_var)

    data = temp[in_vars]     # 只返回原始 X
    labels = temp[out_var]   # 原始 y
    return data, labels      # 不做 fillna / 不做缩放
