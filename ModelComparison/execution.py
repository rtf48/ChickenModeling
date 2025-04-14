import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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

    # Train the model
    model.fit(data_train, labels_train)

    return model

def train_svm(data_train, labels_train, kernel='rbf', C=1.0, epsilon=0.1):
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    model.fit(data_train, labels_train)

    return model




            

