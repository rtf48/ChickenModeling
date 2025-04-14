import matplotlib as plt
import shap
import pandas as pd
import numpy as np

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

def plot_some_interactions(model, data, label):
    
    shap_ivs, manip_ivs = compute_shap(model, data)

        


    important_interactions = manip_ivs.loc[["C14,g","C16:0,g","C18:0,g"],
                                              ["C18:1,g","C18:2 cis n-6 LA,g","C18:3 cis n-3 ALA,g","C22:6n-3 DHA,g"]]
        
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

