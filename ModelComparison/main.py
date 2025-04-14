import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

import analysis
import shapAnalysis
import dataset
import listStorage as ls



def train_rf(data_train, labels_train, n_estimators=100, max_depth=5):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    model.fit(data_train, labels_train)

    return model

def train_lr(data_train, labels_train):
    model = LinearRegression()

    model.fit(data_train, labels_train)

    return model

def train_nn(data_train, labels_train, hidden_layer_sizes=(64, 32), 
             activation='relu', solver='adam', max_iter=100000):
    
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,  # Two hidden layers
                        activation=activation,            # ReLU activation
                        solver=solver,                # Optimizer
                        max_iter=max_iter
                        )
    
    model.fit(data_train, labels_train)

    return model
    

def train_gb(data_train, labels_train, n_estimators=1000, learning_rate=0.1,
              max_depth=5, subsample=0.8, validation_fraction=0.2,
              n_iter_no_change=10, tol=0.0001):

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,        # Number of boosting stages
        learning_rate=learning_rate,       # Learning rate (shrinkage)
        max_depth=max_depth,             # Maximum depth of each tree
        subsample=subsample,           # Fraction of samples used for fitting each tree
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol
    )

    model.fit(data_train, labels_train)

    return model

def train_svm(data_train, labels_train, kernel='rbf', C=1.0, epsilon=0.1):
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    model.fit(data_train, labels_train)

    return model

def compare_models(features, target):

    data, labels = dataset.get_data(features, target)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    rf_model = train_rf(train_data, train_labels)
    lr_model = train_lr(train_data, train_labels)
    svm_model = train_svm(train_data, train_labels)
    nn_model = train_nn(train_data, train_labels)
    gb_model = train_gb(train_data, train_labels)

    models = {'rf':rf_model,'lr':lr_model,'svm':svm_model,'nn':nn_model,'gb':gb_model}
    metrics = {}

    for name, model in models.items():
        individual_metrics = analysis.evaluate(model, test_data, test_labels)
        metrics[name] = individual_metrics

        print(name)
        for k, v in individual_metrics.items():
            print(k, v)

    return metrics

def eval_all(features, targets):

    results = pd.DataFrame(columns=[
    'rf rmse', 'rf r2', 'rf mape',
    'lr rmse', 'lr r2', 'lr mape',
    'svm rmse', 'svm r2', 'svm mape',
    'nn rmse', 'nn r2', 'rnn mape',
    'gb rmse', 'gb r2', 'gb mape'])

    for t in targets:
        metrics = compare_models(features, t)
        results.loc[t] = [metrics['rf']['rmse'],metrics['rf']['r2'],metrics['rf']['mape'],
                          metrics['lr']['rmse'],metrics['lr']['r2'],metrics['lr']['mape'],
                          metrics['svm']['rmse'],metrics['svm']['r2'],metrics['svm']['mape'],
                          metrics['nn']['rmse'],metrics['nn']['r2'],metrics['nn']['mape'],
                          metrics['gb']['rmse'],metrics['gb']['r2'],metrics['gb']['mape']]
        
    return results

def save_output_csv(name, df):
     with open(f'ModelComparison/outputs/{name}.csv', "w") as file:
        file.write(df.to_csv())

compare_all = eval_all(ls.all_features, ls.all_targets)
save_output_csv('compare-all-models', compare_all)

        
    


            

