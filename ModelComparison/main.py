import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import attention

import analysis
import shapAnalysis
import dataset
import listStorage as ls
import models

rf = RandomForestRegressor(
    n_estimators=597, 
    max_depth=5,
    min_samples_split=8,
    min_samples_leaf=5,
    max_features=0.27,
    random_state=42
)

lr = LinearRegression()

nn = MLPRegressor(hidden_layer_sizes=(64, 32),  # Two hidden layers
                        activation='relu',            # ReLU activation
                        solver='adam',                # Optimizer
                        max_iter=100000,
                        early_stopping=True,        # Enables early stopping
                        validation_fraction=0.1,    # Fraction of training data for validation
                        n_iter_no_change=10,        # Stop if no improvement after 10 epochs
                        random_state=42
                        )

gb = GradientBoostingRegressor(
        n_estimators=1000,        # Number of boosting stages
        learning_rate=0.001,       # Learning rate (shrinkage)
        max_depth=34,
        min_samples_leaf=3,
        max_features=0.85,             # Maximum depth of each tree
        subsample=0.9,           # Fraction of samples used for fitting each tree
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=0.0001,
        random_state=42
    )

gb2 = GradientBoostingRegressor(
        n_estimators=1000,        # Number of boosting stages
        learning_rate=0.03,       # Learning rate (shrinkage)
        max_depth=20,
        min_samples_leaf=1,             # Maximum depth of each tree
        subsample=0.9,           # Fraction of samples used for fitting each tree
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=0.0001,
        random_state=42
    )

gb3 = GradientBoostingRegressor(
        n_estimators=1000,        # Number of boosting stages
        learning_rate=0.03,       # Learning rate (shrinkage)
        max_depth=20,
        min_samples_leaf=1,             # Maximum depth of each tree
        subsample=0.9,           # Fraction of samples used for fitting each tree
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=0.0001,
        random_state=42
    )

svm = SVR(kernel='rbf', C=1.0, epsilon=0.1)



def compare_models(features, target):

    data, labels = dataset.get_data(features, target)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    rf_model = rf.fit(train_data, train_labels)
    lr_model = lr.fit(train_data, train_labels)
    svm_model = svm.fit(train_data, train_labels)
    nn_model = nn.fit(train_data, train_labels)
    gb_model = gb.fit(train_data, train_labels)
    att_model = attention.regressor.fit(train_data, train_labels)


    models = {
        #'rf':rf_model,
        #'lr':lr_model,
        #'svm':svm_model,
        'nn':nn_model,
        #'gb':gb_model, ''
        'att':att_model}
    metrics = {}

    for name, model in models.items():
        individual_metrics = analysis.evaluate(model, test_data, test_labels)
        metrics[name] = individual_metrics

        print(target, name)
        for k, v in individual_metrics.items():
            print(k, v)
    
    return metrics

def eval_all(features, targets):

    results = pd.DataFrame(columns=[
    #'rf rmse', 'rf r2', 'rf mape',
    #'lr rmse', 'lr r2', 'lr mape',
    #'svm rmse', 'svm r2', 'svm mape',
    'nn rmse', 'nn r2', 'rnn mape',
    #'gb rmse', 'gb r2', 'gb mape',
    'att rmse', 'att r2', 'att mape'])

    failed_runs = []
    

    for t in targets:

        try:
            metrics = compare_models(features, t)
            results.loc[t] = [
                          #metrics['rf']['rmse'],metrics['rf']['r2'],metrics['rf']['mape'],
                          #metrics['lr']['rmse'],metrics['lr']['r2'],metrics['lr']['mape'],
                          #metrics['svm']['rmse'],metrics['svm']['r2'],metrics['svm']['mape'],
                          metrics['nn']['rmse'],metrics['nn']['r2'],metrics['nn']['mape'],
                          #metrics['gb']['rmse'],metrics['gb']['r2'],metrics['gb']['mape'],
                          metrics['att']['rmse'],metrics['att']['r2'],metrics['att']['mape']]
        except:
            failed_runs += [t]

    if len(failed_runs) > 0:
        print(f"Training failed for {failed_runs}")
        
    return results

def save_output_csv(name, df):
     with open(f'ModelComparison/outputs/{name}.csv', "w") as file:
        file.write(df.to_csv())

#compare_all = eval_all(ls.target_features_comp, ls.target_labels_1 + ls.target_labels_3)
#save_output_csv('attention2', compare_all)
        


data, labels = dataset.get_data(ls.target_features_comp, 'Breast PUFA')
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
model1 = gb.fit(train_data, train_labels)
model2 = models.gb_model.fit(train_data, train_labels)
metrics1 = analysis.evaluate(model1, test_data, test_labels)
metrics2 = analysis.evaluate(model2, test_data, test_labels)
print(metrics1)
print(metrics2)

            

