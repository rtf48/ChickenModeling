from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import cross_val_predict, KFold

def evaluate(model, data, labels):

    predictions = model.predict(data)

    rmse = root_mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)

    metrics = {'RMSE':rmse, 'R2':r2, 'MAPE':mape}

    return metrics

def evaluate_cross_validation(model, data, labels):

    splitter = KFold(n_splits=5, shuffle=True, random_state=24)
    predictions = cross_val_predict(model, data, labels, cv=splitter)

    rmse = root_mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)

    metrics = {'RMSE':rmse, 'R2':r2, 'MAPE':mape}

    return metrics

def evaluate_tree(model):

    explainers = {'Importances':model.feature_importances_}

    return explainers
