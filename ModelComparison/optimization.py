import models
import dataset
import analysis
import listStorage as ls

import sklearn
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import r2_score


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

def optimize_model(model, search_space, features, target, filename=""):

    data, labels = dataset.get_data(features, target)


    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        cv=5,
        scoring='r2',
        n_jobs=1,
        random_state=42
        )

    opt.fit(data, labels)

    # Print best score and parameters
    print(target)
    print("Best R2 score: {:.4f}".format(opt.best_score_))
    print("Best hyperparameters:")
    print(opt.best_params_)

    return(opt.best_params_)

for target in (ls.target_labels_1 + ['Breast SFA','Breast MUFA','Breast PUFA','Breast n-3',
                                    'Breast C18:3','Breast C22:6']):
    

    optimize_model(models.gb_model, gb_search_space, ls.target_features_comp, target)
