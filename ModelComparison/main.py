import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

import analysis
import shapAnalysis
import dataset
import listStorage as ls
import models as m
import attention


def compare_models(features, target):

    data, labels = dataset.get_data(features, target)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    nn_model = m.nn_model.fit(train_data, train_labels)
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

def eval_one_gb(features, targets):

    results = pd.DataFrame(columns=['rmse', 'r2', 'mape'])

    failed_runs = []

    for t in targets:

        data, labels = dataset.get_data(features, t)
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

        
        model = m.get_gb(t)
        model.fit(train_data, train_labels)
        metrics = analysis.evaluate(model, test_data, test_labels)

        original = m.gb_model
        original.fit(train_data, train_labels)
        o_metrics = analysis.evaluate(original, test_data, test_labels)

        if o_metrics['r2'] > metrics['r2']:
            #print('original superior by', o_metrics['r2'] - metrics['r2'], 'for', t)
            metrics = o_metrics
            model = original

        results.loc[t] = [metrics['rmse'],metrics['r2'],metrics['mape']]

        joblib.dump(model, f'ModelComparison/models/{t}_gb.pkl')

        #interactions, readable_interactions = shapAnalysis.compute_shap(model, data, t)
        #save_output_csv(f'shapOutputs/{t}', readable_interactions)


    if len(failed_runs) > 0:
        print(f"Training failed for {failed_runs}")
        
    return results


def save_output_csv(name, df):
     with open(f'ModelComparison/outputs/{name}.csv', "w") as file:
        file.write(df.to_csv())

#compare_all = eval_all(ls.target_features_comp, ls.target_labels_1 + ls.target_labels_3)
#save_output_csv('attention2', compare_all)

#shap_inter_results = eval_one(ls.target_features_comp, ls.all_targets)
#save_output_csv('shap_inter_results', shap_inter_results)

eval_one_gb(ls.target_features_comp, ls.all_targets)


            

