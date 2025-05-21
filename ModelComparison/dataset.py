import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler


raw = pd.read_csv('smallModel/Dataset_3_12.csv')

def set_raw(file):
    global raw
    raw = pd.read_csv(file)

z_norm_outputs = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK', 'liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']



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

    scaler = MinMaxScaler()

    temp = raw.dropna(subset=out_var)

    if out_var in z_norm_outputs:
        temp = z_norm(temp, in_vars)

    data = temp[in_vars]
    labels = temp[out_var]
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data = data.fillna(data.mean())
    data = data.fillna(0) #Necessary incase an entire column is NaN, but shouldn't affect anything
    
    return data, labels 

