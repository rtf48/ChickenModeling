import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import math

dataset = pd.read_csv('smallModel/Calculated_Value_Dataset_Weekly_Encoding.csv')

target_features_comp = ['time period',"ME, kcal/kg",'Overall','NDF','ADF','NFC','Crude fiber',
                        'Starch','CP','Arginine','Histidine','Isoleucine','Leucine','Lysine','Methionine',
                        'Phenylalanine','Threonine','Tryptophan','Valine','Alanine','Aspartic acid',
                        'Cystine','Met + Cys','Glutamic acid','Glycine','Proline','Serine','Tyrosine',
                        'Phe + Tyr','Ether extract','SFA','MUFA','PUFA','n-3 PUFA','n-6 PUFA',
                        'n-3/n-6 ratio','C14','C15:0','C15:1','C16:0','C16:1','C17:0','C17:1','C18:0','C18:1',
                        'C18:2 cis n-6 LA','C18:3 cis n-3 ALA','C20:0','C20:1','C20:4n-6 ARA',
                        'C20:5n-3 EPA','C22:0','C22:1','C22:6n-3 DHA','C24:0','Ash','Vitamin A IU/kg',
                        'beta-carotene','Vitamin D3 IU/kg','Vitamin D3 25-Hydroxyvitamin D',
                        'Vitamin E IU/kg','Vitamin K ppm','AST ppm','Thiamin ppm','Riboflavin ppm',
                        'Niacin ppm','Pantothenic acid ppm','Pyridoxine ppm','Biotin ppm',
                        'Folic acid ppm','Vitamin B12 ppm','Choline ppm','Calcium','Total Phosphorus',
                        'Inorganic available P','Ca:P ratio','Na','Cl','K','Mg','S','Cu ppm','I ppm','Fe','Mn',
                        'Se','Zn']


fatty_acids = ['SFA','MUFA','PUFA','n-3 PUFA','n-6 PUFA','n-3/n-6 ratio','C14','C15:0',
               'C15:1','C16:0','C16:1','C17:0','C17:1','C18:0','C18:1','C18:2 cis n-6 LA',
               'C18:3 cis n-3 ALA','C20:0','C20:1','C20:4n-6 ARA','C20:5n-3 EPA','C22:0','C22:1',
               'C22:6n-3 DHA','C24:0','time period']

target_labels_1 = ['average feed intake g/d','bodyweightgain']

target_labels_2 = ['akp','alt (U/L)','glucose (g/L)',
                 'nefa','pip','tc','tg',
                 'trap','uric acid','BCA']

#The following have too little data: Plasma C16:1, Plasma C18:1, Plasma C18:3, Plasma n-6, 
#Plasma C20:5, Liver C18:1
#'Plasma SFA','Plasma MUFA','Plasma PUFA','Plasma n-3',
target_labels_3 = [
                   'Liver PUFA','Liver n-3','Liver n-6','Liver C18:3 ','Liver C20:5',
                   'Liver C22:6','Breast SFA','Breast MUFA','Breast PUFA','Breast n-3',
                   'Breast n-6','Breast C18:3 ','Breast C20:5','Breast C22:6','Thigh SFA',
                   'Thigh MUFA','Thigh PUFA','Thigh n-3','Thigh n-6','Thigh C18:3 ',
                   'Thigh C20:5','Thigh C22:6']

target_labels_4 = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK','breast LAT1','breast CAT1',
                   'breast SNAT2','breast VDAC1','breast ANTKMT','breast AKT1',
                   'IGF1','IGFR','IRS1','FOXO1','LC3-1','MyoD','MyoG','Pax3',
                   'Pax7','Mrf4','Mrf5','liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']

#target_features_comp = fatty_acids

def train(features, target):

    scaler = MinMaxScaler()

    temp = dataset.dropna(subset=target)
    data = temp[features]
    labels = temp[target]
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data = data.fillna(data.mean())
    #data = data.fillna(0) #Necessary incase an entire column is NaN, but shouldn't affect anything




    # Split the data into training and testing sets
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)


    # Initialize the Gradient Boosting Regressor
    model = LinearRegression()

    # Train the model
    model.fit(data_train, labels_train)

    # Make predictions
    pred_train = model.predict(data_train)
    pred_test = model.predict(data_test)

    residuals = labels_train - pred_train

    n = len(labels_train)
    sigma_squared = np.var(residuals)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - np.sum(residuals**2) / (2 * sigma_squared)


    # Evaluate the model
    k = len(features)
    train_rmse = math.sqrt(mean_squared_error(labels_train, pred_train))
    train_r2 = r2_score(labels_train, pred_train)
    train_mape = mean_absolute_percentage_error(labels_train, pred_train)
    train_aic = 2 * k - 2 * (-n * log_likelihood)
    train_bic = k * math.log(n) - 2 * (-n * log_likelihood)

    test_rmse = math.sqrt(mean_squared_error(labels_test, pred_test))
    test_r2 = r2_score(labels_test, pred_test)
    test_mape = mean_absolute_percentage_error(labels_test, pred_test)
    
    print(f'''{target} evaluation:
    Training RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, MAPE: {train_mape:.4f}
    Testing RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, MAPE: {test_mape:.4f}''')
    
    metrics = pd.DataFrame({
        'Metric':['Training RMSE','Training R2','Training MAPE','Testing RMSE','Testing R2','Testing MAPE'],
        'Value':[train_rmse, train_r2, train_mape, test_rmse, test_r2, test_mape]
    })

    return metrics

def evaluate(targets):
    for i in targets:
        metrics = train(target_features_comp, i)
        
        
        metric_string = f'''{i} evaluation:
        Training RMSE: {metrics['Value'][0]:.4f}, R2: {metrics['Value'][1]:.4f}, MAPE: {metrics['Value'][2]:.4f}
        Testing RMSE: {metrics['Value'][3]:.4f}, R2: {metrics['Value'][4]:.4f}, MAPE: {metrics['Value'][5]:.4f}'''

        with open(f'smallModel/outputs/{i}.txt', "w") as file:
            file.write(metric_string)
            
def fill_csv(name, inputs):

    metric_frame = pd.DataFrame({'Metric':['Training RMSE','Training R2',
                                           'Training MAPE','Testing RMSE',
                                           'Testing R2','Testing MAPE']})

    targets = target_labels_1 + target_labels_2 + target_labels_3 + target_labels_4

    for label in targets:
        metrics = train(inputs, label)
        metric_frame[label] = metrics['Value']

    with open(f'smallModel/outputs/{name}_metrics.csv', "w") as file:
        file.write(metric_frame.to_csv())


#evaluate(target_labels_1)
#evaluate(target_labels_2)
#evaluate(target_labels_3)
#evaluate(target_labels_4)

fill_csv('linear_regression', fatty_acids)