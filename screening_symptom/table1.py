import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

dataframe_image = pd.read_csv('/home/jaehan0605/MAIN_ASDfundus/age_sex_matched_df.csv')

ASD_full_image = dataframe_image[dataframe_image['data_class']=='ASD']
ASD_full_patient = ASD_full_image.drop_duplicates(subset=['patient_id'])

ASD_ADOS_severe_patient = ASD_full_patient[ASD_full_patient['ados']>=8]
ASD_ADOS_nonsevere_patient = ASD_full_patient[ASD_full_patient['ados']<8]

ASD_SRS_severe_patient = ASD_full_patient[ASD_full_patient['srs']>=76]
ASD_SRS_nonsevere_patient = ASD_full_patient[ASD_full_patient['srs']<76]

TD_full_image = dataframe_image[dataframe_image['data_class']=='TD']
TD_full_patient = TD_full_image.drop_duplicates(subset=['patient_id'])

for purpose in ['TD','ADOS','SRS']:
    if purpose == 'TD':
        print("TD vs ASD")
        data1 = TD_full_patient
        data2 = ASD_full_patient
    elif purpose == 'ADOS':
        print("ADOS>=8 vs <8")
        data1 = ASD_ADOS_severe_patient
        data2 = ASD_ADOS_nonsevere_patient
    elif purpose == 'SRS':
        print("SRS>=76 vs <76")
        data1 = ASD_SRS_severe_patient
        data2 = ASD_SRS_nonsevere_patient

    for variable in ['age', 'ados', 'srs', 'sex']:
        if variable in ['age', 'ados', 'srs']:
            data1_mean = round(data1[f'{variable}'].mean(),2)
            data1_sd = round(data1[f'{variable}'].std(),2)
            data2_mean = round(data2[f'{variable}'].mean(),2)
            data2_sd = round(data2[f'{variable}'].std(),2)

            t_statistic, p_value = ttest_ind(list(data1[f'{variable}'].dropna()), list(data2[f'{variable}'].dropna()))

            p = round(p_value, 4)

            if purpose == 'TD':
                print(f"TD {variable} : {data1_mean} ({data1_sd}), ASD: {data2_mean} ({data2_sd}), P = {p}")
            else:
                print(f"{variable} severe: {data1_mean} ({data1_sd}), non-severe: {data2_mean} ({data2_sd}), P = {p}")
        
        else: 
            data1_M = int(pd.DataFrame(data1['sex'].value_counts()).loc['M'])
            data1_F = int(pd.DataFrame(data1['sex'].value_counts()).loc['F'])
            data2_M = int(pd.DataFrame(data2['sex'].value_counts()).loc['M'])
            data2_F = int(pd.DataFrame(data2['sex'].value_counts()).loc['F'])
            tab_data = [[data1_M, data1_F], [data2_M, data2_F]]
            chi2, p_value, dof, expected = chi2_contingency(tab_data)
            p = round(p_value, 4)

            if purpose == 'TD':
                print(f"TD {variable} : Male {data1_M}/ Female {data1_F}, ASD: Male {data2_M}/ Female {data2_F}, P = {p}")
            else:
                print(f"{variable} severe: Male {data1_M}/ Female {data1_F}, non-severe: Male {data2_M}/ Female {data2_F}, P = {p}")

