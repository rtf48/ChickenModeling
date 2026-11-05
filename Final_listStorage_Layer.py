z_norm_features = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK', 'liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']

target_features_comp = ["Start", "End", "Window Length", "C14,g",
    "C16:0,g",
    "C16:1,g",
    "C18:0,g",
    "C18:1,g",
    "C18:2 cis n-6 LA,g",
    "C18:3 cis n-3 ALA,g",
    "C20:4n-6 ARA,g",
    "C20:5n-3 EPA,g",
    "C22:6n-3 DHA,g"
]

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


target_labels_5 = [
    'Egg weight', 'Laying rate',
    'Egg C14', 'Egg C15', 'Egg C15:1', 'Egg C16', 'Egg C16:1',
    'Egg C17', 'Egg C17:1', 'Egg C18', 'Egg C18:1',
    'Egg C18:2', 'Egg C18:3 n-6', 'Egg C18:3 n-3',
    'Egg C20:4', 'Egg C20:5', 'Egg C22:5', 'Egg C22:6',
    'Egg SFA', 'Egg MUFA', 'Egg PUFA', 'Egg n-6', 'Egg n-3'
]


all_targets = target_labels_1 + target_labels_2 + target_labels_3 + target_labels_4

not_enough_data = ['Plasma C16:1', 'Plasma C18:1', 'Plasma C18:3', 'Plasma C20:5', 'Liver C18:1', 'Plasma MUFA',]

