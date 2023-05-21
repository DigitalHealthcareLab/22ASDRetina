import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

df_ASD = pd.read_csv('/home/jaehan0605/MAIN_ASDfundus/Result_simpleRGB_ASD/Macular_centred/Macular_Zone_C_Measurement.csv')
df_TD = pd.read_csv('/home/jaehan0605/MAIN_ASDfundus/Result_simpleRGB_TD/Macular_centred/Macular_Zone_C_Measurement.csv')

df_ASD.columns[2:]

list_for_df = []

for feature in df_ASD.columns[2:]:
    ASD_feature = df_ASD[f'{feature}']
    ASD_feature = list(ASD_feature)
    ASD_feature_list = []
    for j in ASD_feature:
        if j > 0:
            ASD_feature_list.append(j)
        else: pass
            
    TD_feature = df_TD[f'{feature}']
    TD_feature = list(TD_feature)
    TD_feature_list = []
    for j in TD_feature:
        if j > 0:
            TD_feature_list.append(j)
        else: pass
    
    t_statistic, p_value = ttest_ind(ASD_feature_list, TD_feature_list)

    ASD_feature_np = np.array(ASD_feature_list)
    TD_feature_np = np.array(TD_feature_list)

    print(f"<{feature}>")
    print(f"ASD = mean {round(ASD_feature_np.mean(),5)} ({round(ASD_feature_np.std(),5)})")
    print(f"TD = mean {round(TD_feature_np.mean(),5)} ({round(TD_feature_np.std(),5)})")
    print(f"P = {p_value}") 

