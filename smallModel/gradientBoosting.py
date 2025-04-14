import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error
import pandas as pd
import math
import shap

data_set = pd.read_csv('smallModel/Dataset_3_12.csv')

z_norm_features = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK', 'liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']

target_features_comp = ['time period',"ME, kcal","NDF,g","ADF,g",
                        "NFC,g","Crude fiber,g","Starch,g","CP,g","Arginine,g",
                        "Histidine,g","Isoleucine,g","Leucine,g","Lysine,g",
                        "Methionine,g","Phenylalanine,g","Threonine,g",
                        "Tryptophan,g","Valine,g","Alanine,g","Aspartic acid,g",
                        "Cystine,g","Met + Cys,g","Glutamic acid,g","Glycine,g",
                        "Proline,g","Serine,g","Tyrosine,g","Phe + Tyr,g",
                        "Ether extract,g","SFA,g","MUFA,g","PUFA,g","n-3 PUFA,g",
                        "n-6 PUFA,g","n-3:n-6 ratio,g","C14,g","C15:0,g","C15:1,g",
                        "C16:0,g","C16:1,g","C17:0,g","C17:1,g","C18:0,g","C18:1,g",
                        "C18:2 cis n-6 LA,g","C18:3 cis n-3 ALA,g","C20:0,g",
                        "C20:1,g","C20:4n-6 ARA,g","C20:5n-3 EPA,g","C22:0,g",
                        "C22:1,g","C22:5,g","C22:6n-3 DHA,g","C24:0,g","Ash,mg",
                        "Vitamin A IU per kg","beta-carotene,mg","Vitamin D3 IU per kg",
                        "Vitamin D3 25-Hydroxyvitamin D, IU","Vitamin E IU per kg","Vitamin K ppm",
                        "AST ppm","Thiamin ppm","Riboflavin ppm","Niacin ppm",
                        "Pantothenic acid ppm","Pyridoxine ppm","Biotin ppm",
                        "Folic acid ppm","Vitamin B12 ppm","Choline ppm","Calcium,g",
                        "Total Phosphorus,g","Inorganic available P,g","Ca:P ratio",
                        "Na,mg","Cl,mg","K,mg","Mg,mg","S,mg","Cu ppm","I ppm","Fe,mg",
                        "Mn,mg","Se,mg","Zn,mg"]
#the following have no data and should not be included in any dataset:
#overall, c15:0, c15:1, c17:0, c17:1, Vitamin D3 25-Hydroxyvitamin D, 



fatty_acids = ["SFA,g","MUFA,g","PUFA,g","n-3 PUFA,g",
                        "n-6 PUFA,g","n-3:n-6 ratio,g","C14,g",
                        "C16:0,g","C16:1,g","C18:0,g","C18:1,g",
                        "C18:2 cis n-6 LA,g","C18:3 cis n-3 ALA,g","C20:0,g",
                        "C20:1,g","C22:0,g",
                        "C22:6n-3 DHA,g","C24:0,g", "Start", "End"]
# removed c15:0, c15:1, c17:0, c17:1, C20:4n6, C20:5n3, C22:1, C22:5 due to lack of data

minerals = ["Calcium,g",
                        "Total Phosphorus,g","Inorganic available P,g","Ca:P ratio",
                        "Na,g","Cl,g","K,g","Mg,g","S,mg","Cu mg","I mg","Fe,mg",
                        "Mn,mg","Se,mg","Zn,mg", "Start", "End"]
#is ash a mineral? is choline?


vitamins = ["Vitamin A IU","beta-carotene,mg","Vitamin D3 IU",
                        "Vitamin E IU","Vitamin K mg",
                        "Thiamin mg","Riboflavin mg","Niacin mg",
                        "Pantothenic acid mg","Pyridoxine mg","Biotin mg",
                        "Folic acid mg","Vitamin B12 mg ", "Start", "End"]
#removed Vitamin D3 25-Hydroxyvitamin D due to lack of data

target_labels_1 = ['average feed intake g per d','bodyweightgain,g']

target_labels_2 = ['akp U per ml','alt (U per L)','glucose (g per L)',"nefa,umol per L",
                   'pip mg per dL','tc mg per g','tg mg per g','trap U per L','uric acid mmol per L','BCA']

#The following have too little data: Plasma C16:1, Plasma C18:1, Plasma C18:3, Plasma n-6, 
#Plasma C20:5, Liver C18:1
#'Plasma SFA','Plasma MUFA','Plasma PUFA','Plasma n-3','Breast C20:5','Liver C20:5',
target_labels_3 = [
                   'Liver PUFA','Liver n-3','Liver n-6','Liver C18:3 ',
                   'Liver C22:6','Breast SFA','Breast MUFA','Breast PUFA','Breast n-3',
                   'Breast n-6','Breast C18:3 ','Breast C22:6','Thigh SFA',
                   'Thigh MUFA','Thigh PUFA','Thigh n-3','Thigh n-6','Thigh C18:3',
                   'Thigh C20:4','Thigh C22:6']

target_labels_4 = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK','breast LAT1','breast CAT1',
                   'breast SNAT2','breast VDAC1','breast ANTKMT','breast AKT1',
                   'IGF1','IGFR','IRS1','FOXO1','LC3-1','MyoD','MyoG','Pax3',
                   'Pax7','Mrf4','Mrf5','liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']




def z_norm(dataset, target):
    data_dha = dataset[dataset['Study'] == 'broiler chicken DHA'].copy()
    data_algae = dataset[dataset['Study'] == 'broiler microalgae'].copy()
    remainder = dataset[dataset['Study'] != 'broiler microalgae'].copy()
    remainder = remainder[remainder['Study'] != 'broiler chicken DHA'].copy()

    z_norm = StandardScaler()

    data_dha[target] = z_norm.fit_transform(data_dha[[target]])
    data_algae[target] = z_norm.fit_transform(data_algae[[target]])

    return pd.concat([data_dha, data_algae, remainder])

def split_data(features, target, raw_data, random=42):
    scaler = MinMaxScaler()
    poly = PolynomialFeatures(degree=2, include_bias=True)

    temp = raw_data.dropna(subset=target)

    #if target in z_norm_features:
    #    temp = z_norm(temp, target)

    data = temp[features]
    labels = temp[target]
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    #ptransform_data = pd.DataFrame(scaler.fit_transform(poly.fit_transform(data)), columns=poly.get_feature_names_out(data.columns))
    #data = ptransform_data
    data = data.fillna(data.mean())
    #data = data.fillna(0) #Necessary incase an entire column is NaN, but shouldn't affect anything

    return data, labels



def train(data_train, labels_train):

    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=1000,        # Number of boosting stages
        learning_rate=0.1,       # Learning rate (shrinkage)
        max_depth=5,             # Maximum depth of each tree
        subsample=0.8,           # Fraction of samples used for fitting each tree
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=0.0001
    )

    # Train the model
    model.fit(data_train, labels_train)

    return model



def eval_all(model, data, labels, data_train, data_test, labels_train, labels_test, target):
    # Make predictions
    pred_train = model.predict(data_train)
    pred_test = model.predict(data_test)

    splitter = KFold(n_splits=5, shuffle=True, random_state=24)
    pred_cv = cross_val_predict(model, data, labels, cv=splitter)
    
    # Evaluate the model

    residuals = labels_train - pred_train

    n = len(labels_train)
    sigma_squared = np.var(residuals)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - np.sum(residuals**2) / (2 * sigma_squared)

    #MAPE doesn't like 0 values
    #labels_train =  [0.001 if x == 0 else x for x in labels_train]
    #labels_test =  [0.001 if x == 0 else x for x in labels_test]

    train_rmse = root_mean_squared_error(labels_train, pred_train)
    train_r2 = r2_score(labels_train, pred_train)
    train_mape = mean_absolute_percentage_error(labels_train, pred_train)
    

    test_rmse = root_mean_squared_error(labels_test, pred_test)
    test_r2 = r2_score(labels_test, pred_test)
    test_mape = mean_absolute_percentage_error(labels_test, pred_test)

    cv_rmse = root_mean_squared_error(labels, pred_cv)
    cv_r2 = r2_score(labels, pred_cv)
    cv_mape = mean_absolute_percentage_error(labels, pred_cv)
    
    print(f'''{target} evaluation:
    Training RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, MAPE: {train_mape:.4f}
    Testing RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, MAPE: {test_mape:.4f}
    Cross Validation RMSE: {cv_rmse:.4f}, R2: {cv_r2:.4f}, MAPE: {cv_mape:.4f}''')
    
    metrics = pd.DataFrame({
        'Metric':['Training RMSE', 'Training R2', 'Training MAPE',
                   'Testing RMSE',' Testing R2','Testing MAPE',
                   'Cross Validation RMSE','Cross Validation R2','Cross Validation MAPE'],
        'Value':[train_rmse, train_r2, train_mape, 
        test_rmse, test_r2, test_mape,
        cv_rmse, cv_r2, cv_mape]
    })

    importance = pd.DataFrame({
        'Feature': data.columns,
        'Importance': model.feature_importances_
    })

    return metrics, importance

def compute_shap(model, data):

    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    #bs = shap.plots.beeswarm(shap_values, max_display=30, show=False)
    #bs = shap.summary_plot(shap_values, data.iloc[:2000,:], max_display=30, show=False)
    #bs = shap.dependence_plot(
    #("C14,g", "C22:6n-3 DHA,g"),
    #shap_values, data.iloc[:2000,:], 
    #show=False)

    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data)
    siv_sum = shap_interaction_values.sum(0)

    shap_ivs = pd.DataFrame(siv_sum, columns = data.columns, index = data.columns)

    siv_manip = np.abs(shap_interaction_values).sum(0)
    for i in range(siv_manip.shape[0]):
            siv_manip[i,i] = 0
    siv_manip = siv_manip/siv_manip.sum()
    
    manip_ivs = pd.DataFrame(siv_manip, columns = data.columns, index = data.columns)

    


    

    #tmp = np.abs(shap_interaction_values).sum(0)
    #for i in range(tmp.shape[0]):
    #        tmp[i,i] = 0
    #inds = np.argsort(-tmp.sum(0))[:50]
    #tmp2 = tmp[inds,:][:,inds]
    #plt.figure(figsize=(12,12))
    #plt.imshow(tmp2)
    #plt.yticks(range(tmp2.shape[0]), data.columns[inds], rotation=50.4, horizontalalignment="right")
    #plt.xticks(range(tmp2.shape[0]), data.columns[inds], rotation=50.4, horizontalalignment="left")
    #plt.gca().xaxis.tick_top()

    #plt.tight_layout()
    #plt.savefig(f'smallModel/outputs/plots/{target}')
    #plt.close()

    return shap_interaction_values, manip_ivs



    
    
            
def fill_csv(name, inputs, targets):

    importance_frame = pd.DataFrame({'Variable':inputs})
    metric_frame = pd.DataFrame({'Metric':['Training RMSE','Training R2',
                                           'Training MAPE','Testing RMSE',
                                           'Testing R2','Testing MAPE']})

    
    for label in targets:

        data, labels = split_data(inputs, label, data_set)
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        model = train(data_train, labels_train)

        metrics, importance = eval_all(model, data, labels, data_train, data_test, labels_train, labels_test, label)
        importance_frame[label] = importance['Importance']
        metric_frame[label] = metrics['Value']

        shap_ivs, manip_ivs = compute_shap(model, data)

        


        important_interactions = manip_ivs.loc[["C14,g","C16:0,g","C18:0,g"],
                                              ["C18:1,g","C18:2 cis n-6 LA,g","C18:3 cis n-3 ALA,g","C22:6n-3 DHA,g"]]
        
        if metrics["Value"][4] > 1:
            #print(important_interactions)
            for i in important_interactions.columns:
                for j in important_interactions.index:
                    if important_interactions[i][j] > 0.025:
                        print(important_interactions[i][j])
                        fig, ax = plt.subplots()
                        plt.imshow(manip_ivs)
                        #for i in range(manip_ivs.shape[0]):
                        #    for j in range(manip_ivs.shape[1]):
                        #        ax.text(j, i, f'{manip_ivs.iloc[i, j]:.2f}', ha='center', va='center', color='white',fontsize=6)
                        plt.yticks(range(manip_ivs.shape[0]), data.columns, rotation=50.4, horizontalalignment="right")
                        plt.xticks(range(manip_ivs.shape[0]), data.columns, rotation=50.4, horizontalalignment="left")
                        plt.gca().xaxis.tick_top()
                        ax.set_title(label)
                        fig.tight_layout()

                        plt.savefig(f'smallModel/outputs/plots/{label}_heatmap')
                        plt.close()

                        plot_shap_interaction(shap_ivs, i, j, data, label)
                        
                        
                        

    

    with open(f'smallModel/outputs/{name}_importances.csv', "w") as file:
        file.write(importance_frame.to_csv())
    with open(f'smallModel/outputs/{name}_metrics.csv', "w") as file:
        file.write(metric_frame.to_csv())

def plot_shap_interaction(shap_interaction_values, feature_1, feature_2, data, label):
    """
    Plot SHAP interaction values with color representing magnitude.

    Parameters:
    shap_interaction_values (numpy.ndarray or pd.DataFrame): SHAP interaction values matrix
    feature_1 (str): Name of the first feature (x-axis)
    feature_2 (str): Name of the second feature (y-axis)
    data (pd.DataFrame): Original data used for SHAP interpretation
    """
    
    # Get feature indices
    feature_1_index = data.columns.get_loc(feature_1)
    feature_2_index = data.columns.get_loc(feature_2)

    # Extract interaction values
    interaction_values = shap_interaction_values[:, feature_1_index, feature_2_index]

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[feature_1], data[feature_2], c=np.abs(interaction_values), cmap='viridis', s=60)
    plt.colorbar(scatter, label='Interaction Magnitude')
    
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title(f'SHAP Interaction {label}]: {feature_1} vs {feature_2}')
    plt.grid(True)
    plt.savefig(f'smallModel/outputs/plots/{label}_{feature_1}_vs_{feature_2}')
    plt.close()

valuable_outputs = ['bodyweightgain,g','average feed intake g per d','Liver n-3',
          'Liver C22:6','Breast n-3','Breast C18:3 ','Breast C22:6',
          'Thigh C20:4','Thigh C22:6','breast AMPK','breast mTOR']

targets = target_labels_1 + target_labels_2 + target_labels_3 + target_labels_4

mystery_features = ['breast mTOR','breast MURF1','breast AMPK','liver mTOR','liver MURF1','liver AMPK']
tfa_outputs = [
                   'Liver PUFA','Liver n-3','Liver n-6',
                   'Liver C22:6','Breast SFA','Breast MUFA','Breast PUFA','Breast n-3',
                   'Breast n-6','Breast C18:3 ','Breast C22:6','Thigh SFA',
                   'Thigh MUFA','Thigh PUFA','Thigh n-3','Thigh n-6','Thigh C18:3',
                   'Thigh C20:4','Thigh C22:6']

#run(target_labels_1)
#run(target_labels_2)
#run(target_labels_3)
#run(target_labels_4)

#run(['Thigh C22:6'])

#run(valuable_outputs)
#possibly nonfunctional: breast ampk, breast mtor, breast c18:3, 

fill_csv('technical', fatty_acids, [targets])


