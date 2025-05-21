import models
import dataset
import analysis
import listStorage as ls

import sklearn
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


rf_search_space = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0, prior='uniform')
}

gb_search_space = {
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0, prior='uniform'),
    'learning_rate':Real(0.001, 0.1, prior='uniform'),
    'subsample': Real(0.5, 0.9, prior='uniform'),        
}

nn_search_space = {
    'validation_fraction': Real(0.1, 0.5, prior='uniform'),
    'n_iter_no_change': Integer(5,20),
    'tol': Real(0.0001, 0.01, prior='uniform')
}

def optimize_model(model, search_space, features, target):

    data, labels = dataset.get_data(features, target)


    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        cv=5,
        scoring='r2',
        random_state=42,
        n_iter=100
        )

    opt.fit(data, labels)

    # Print best score and parameters
    print(target)
    print("Best score: {:.4f}".format(opt.best_score_))
    print("Best hyperparameters:")
    print(opt.best_params_)

    return(opt.best_params_)

def optimize_all(model, search_space, features, targets, filename):

    params = pd.DataFrame(columns=['0','1','2','3','4','5'])
    failures = []

    for target in targets:
        try:
            p = optimize_model(model, search_space, features, target)
            params.columns = p.keys()
            params.loc[target] = p.values()
        except:
            failures += [target]
        
    print('optimization failed for:', failures)
    with open(f'ModelComparison/modelParameters/{filename}.csv', "w") as file:
        file.write(params.to_csv())

def get_top_n_features(model, n, features=ls.target_features_comp):
    
    importances = model.feature_importances_

    df = pd.DataFrame({
        'feature': features,
        'importance': importances
    })

    top_features = df.sort_values(by="importance", ascending=False).head(n).reset_index(drop=True)
    return top_features

def feature_selection(model, full_features, target):

    data, labels = dataset.get_data(full_features, target)
    baseline_model = clone(model)
    baseline_model.fit(data, labels)
    splitter = KFold(n_splits=5, shuffle=True, random_state=24)
    baseline_pred = cross_val_predict(model, data, labels, cv=splitter)
    baseline_mse = mean_squared_error(baseline_pred, labels)

    best_accuracy = 0
    best_num_features = 0
    num_features = 4
    while num_features < 50:
        num_features += 1
        select_features = get_top_n_features(baseline_model, num_features)['feature']
        select_data, select_labels = dataset.get_data(select_features, target)
        model.fit(select_data, select_labels)
        select_pred = cross_val_predict(model, select_data, select_labels, cv=splitter)
        select_mse = mean_squared_error(select_pred, select_labels)
        accuracy = baseline_mse/select_mse
        print(target, num_features, accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_features = num_features
    
    return get_top_n_features(baseline_model, best_num_features)['feature']

def find_all_features_gb(full_features, targets, filename):
    
    result = pd.DataFrame(columns=targets, index=range(len(ls.target_features_comp)))

    for target in targets:
        features = feature_selection(models.get_gb(target), full_features, target)
        result[target] = features
    
    with open(f'ModelComparison/outputs/{filename}_features.csv', "w") as file:
        file.write(result.to_csv())

    return result
    
#optimize_all(models.gb_model, gb_search_space, ls.target_features_comp, ls.all_targets, 'new_gb_params')
find_all_features_gb(ls.target_features_comp, ls.accurate_models, 'filtered_models')
