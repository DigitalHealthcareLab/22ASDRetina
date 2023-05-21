import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

def split_dataset_undersampling(save_path: Path, save_path1: Path, save_path2: Path):
    image_paths1 = np.array(list(save_path1.rglob('*.png')))
    image_df1 = pd.DataFrame({'image_path' : image_paths1})
    image_paths2 = np.array(list(save_path2.rglob('*.png')))
    image_df2 = pd.DataFrame({'image_path' : image_paths2})
    
    image_df = pd.concat([image_df1, image_df2], axis = 0)

    image_df['data_class'] = image_df['image_path'].apply(lambda x : x.parent.name)
    image_df['patient_id'] = image_df['image_path'].apply(lambda x : x.stem)
    image_df['patient_name'] = image_df['patient_id'].apply(lambda x: x.split('_')[1])

    patient_df = image_df[['data_class', 'patient_id']]
    
    #randomly undersample
    patient_df.loc[:,'patient_name'] = patient_df['patient_id'].apply(lambda x : x.split('_')[1])
    patient_unique_df = patient_df.copy()
    patient_unique_df = patient_unique_df.loc[:,['patient_name','data_class']].drop_duplicates()

    rus = RandomUnderSampler(sampling_strategy = 'auto', random_state=2023, replacement=False) ##2023
    patient_unique_df, y = rus.fit_resample(patient_unique_df, patient_unique_df['data_class'])

    #split train:test at the patient level
    train_patient_ids, test_patient_ids = train_test_split(patient_unique_df['patient_name'].values, test_size = 0.15, random_state=2022, stratify = patient_unique_df['data_class'].values, shuffle= True) ##2023 for 10%, 2022 for 15%, 2022 for deep ensemble

    X_train = patient_df.query('patient_name in @train_patient_ids')
    X_test = patient_df.query('patient_name in @test_patient_ids')

    X_train.loc[:,'patient_name'] = X_train['patient_id'].apply(lambda x : x.split('_')[1])
    X_train_unique_df = X_train.copy()
    X_train_unique_df = X_train_unique_df.loc[:,['patient_name','data_class']].drop_duplicates()

    #generate 10-fold at the patient level
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2022) #2023 for 10%, 2022 for 15%, 2022 for Deep ensemble
    for fold_num, (tr_patient_ids, val_patient_ids) in enumerate(skf.split(X_train_unique_df['patient_name'], X_train_unique_df['data_class'])) :
        X_train_train = X_train_unique_df.iloc[tr_patient_ids]
        X_val_train = X_train_unique_df.iloc[val_patient_ids]

        image_df[f'FOLD_{fold_num}'] = image_df['patient_name'].apply(lambda x : 0 if x in list(X_train_train['patient_name'])
                                                                                else 1 if x in list(X_val_train['patient_name'])
                                                                                else 2 if x in list(X_test['patient_name'])
                                                                                else 999)   
    image_df.to_csv(save_path.joinpath('dataset.csv'), index=False)


def make_dataframe_OOD(save_path: Path):
    image_path = np.array(list(save_path.rglob('*.png')) + list(save_path.rglob('*.jpg')) + list(save_path.rglob('*.jpeg')) + list(save_path.rglob('*.JPG')))
    image_df = pd.DataFrame({'image_path' : image_path})
    image_df['FOLD_0'] = 2
    image_df.to_csv(save_path.joinpath('dataset.csv'), index=False)


def split_dataset_Kfold(save_path: Path, save_path1: Path, save_path2: Path):
    image_paths1 = np.array(list(save_path1.rglob('*.png')))
    image_df1 = pd.DataFrame({'image_path' : image_paths1})
    image_paths2 = np.array(list(save_path2.rglob('*.png')))
    image_df2 = pd.DataFrame({'image_path' : image_paths2})
    
    image_df = pd.concat([image_df1, image_df2], axis = 0)

    image_df['data_class'] = image_df['image_path'].apply(lambda x : x.parent.name)
    image_df['patient_id'] = image_df['image_path'].apply(lambda x : x.stem)
    image_df['patient_name'] = image_df['patient_id'].apply(lambda x: x.split('_')[1])

    patient_df = image_df[['data_class', 'patient_id']]
    
    #randomly undersample
    patient_df.loc[:,'patient_name'] = patient_df['patient_id'].apply(lambda x : x.split('_')[1])
    patient_unique_df = patient_df.copy()
    patient_unique_df = patient_unique_df.loc[:,['patient_name','data_class']].drop_duplicates()

    rus = RandomUnderSampler(sampling_strategy = 'auto', random_state=2023, replacement=False) ##2023
    patient_unique_df, y = rus.fit_resample(patient_unique_df, patient_unique_df['data_class'])

    #split train:test at the patient level
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2022) 

    for K_fold_num, (tr_patient_ids_K, test_patient_ids_K) in enumerate(skf.split(patient_unique_df['patient_name'], patient_unique_df['data_class'])):

        X_train = patient_unique_df.iloc[tr_patient_ids_K]
        X_test = patient_unique_df.iloc[test_patient_ids_K]

        # X_train.loc[:,'patient_name'] = X_train['patient_id'].apply(lambda x : x.split('_')[1])
        X_train_unique_df = X_train.copy()
        X_train_unique_df = X_train_unique_df.loc[:,['patient_name','data_class']].drop_duplicates()
        
        for fold_num, (tr_patient_ids, val_patient_ids) in enumerate(skf.split(X_train_unique_df['patient_name'], X_train_unique_df['data_class'])) :
            X_train_train = X_train_unique_df.iloc[tr_patient_ids]
            X_val_train = X_train_unique_df.iloc[val_patient_ids]
            image_df[f'FOLD_{K_fold_num}_{fold_num}'] = image_df['patient_name'].apply(lambda x : 0 if x in list(X_train_train['patient_name'])
                                                                                else 1 if x in list(X_val_train['patient_name'])
                                                                                else 2 if x in list(X_test['patient_name'])
                                                                                else 999)               
    
    image_df.to_csv(save_path.joinpath('dataset.csv'), index=False)

