import models
import dataset
import analysis
import listStorage as ls

import sklearn
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import r2_score
import pandas as pd


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
        n_iter=32
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

optimize_all(models.gb_model, gb_search_space, ls.target_features_comp, ls.all_targets, 'gb_params')
#optimize_all(models.gb_model, gb_search_space, ls.target_features_comp, ls.target_labels_3+ls.target_labels_4, 'gb_params')
