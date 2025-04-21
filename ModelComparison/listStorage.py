z_norm_features = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK', 'liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']

target_features_comp = ['Start','End',"ME, kcal","NDF,g","ADF,g","NFC,g","Crude fiber,g",
                "Starch,g","CP,g","Arginine,g","Histidine,g","Isoleucine,g",
                "Leucine,g","Lysine,g","Methionine,g","Phenylalanine,g",
                "Threonine,g","Tryptophan,g","Valine,g","Alanine,g",
                "Aspartic acid,g","Cystine,g","Met + Cys,g","Glutamic acid,g",
                "Glycine,g","Proline,g","Serine,g","Tyrosine,g","Phe + Tyr,g",
                "Ether extract,g","SFA,g","MUFA,g","PUFA,g","n-3 PUFA,g",
                "n-6 PUFA,g","n-3:n-6 ratio,g","C14,g","C15:0,g","C15:1,g",
                "C16:0,g","C16:1,g","C17:0,g","C17:1,g","C18:0,g","C18:1,g",
                "C18:2 cis n-6 LA,g","C18:3 cis n-3 ALA,g","C20:0,g","C20:1,g",
                "C20:4n-6 ARA,g","C20:5n-3 EPA,g","C22:0,g","C22:1,g",
                "C22:6n-3 DHA,g","C24:0,g","Ash,mg",'Vitamin A IU',
                "beta-carotene,mg",'Vitamin D3 IU',
                "Vitamin D3 25-Hydroxyvitamin D, IU",'Vitamin E IU',
                'Vitamin K mg','AST ppm','Thiamin mg','Riboflavin mg','Niacin mg',
                'Pantothenic acid mg','Pyridoxine mg','Biotin mg','Folic acid mg',
                'Vitamin B12 mg ','Choline mg',"Calcium,g","Total Phosphorus,g",
                "Inorganic available P,g",'Ca:P ratio',"Na,g","Cl,g","K,g","Mg,g",
                "S,mg",'Cu mg','I mg',"Fe,mg","Mn,mg","Se,mg","Zn,mg"]

target_labels_1 = ['average feed intake g per d','bodyweightgain,g']

target_labels_2 = ['akp U per ml','alt (U per L)','glucose (g per L)',"nefa,umol per L",
                   'pip mg per dL','tc mg per g','tg mg per g','trap U per L','uric acid mmol per L','BCA']



target_labels_3 = [
                   'Liver PUFA','Liver n-3','Liver n-6','Liver C18:3 ',
                   'Liver C22:6','Breast SFA','Breast MUFA','Breast PUFA','Breast n-3',
                   'Breast n-6','Breast C18:3 ','Breast C22:6','Thigh SFA',
                   'Thigh MUFA','Thigh PUFA','Thigh n-3','Thigh n-6','Thigh C18:3',
                   'Thigh C20:4','Thigh C22:6','Plasma n-6','Plasma SFA','Plasma PUFA',
                    'Plasma n-3','Breast C20:5','Liver C20:5']

target_labels_4 = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK','breast LAT1','breast CAT1',
                   'breast SNAT2','breast VDAC1','breast ANTKMT','breast AKT1',
                   'IGF1','IGFR','IRS1','FOXO1','LC3-1','MyoD','MyoG','Pax3',
                   'Pax7','Mrf4','Mrf5','liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']


all_targets = target_labels_1 + target_labels_2 + target_labels_3 + target_labels_4

not_enough_data = ['Plasma C16:1', 'Plasma C18:1', 'Plasma C18:3', 'Plasma C20:5', 'Liver C18:1', 'Plasma MUFA',]

