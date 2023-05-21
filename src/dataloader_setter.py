from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import CustomDataset

class Dataloader_Setter :

    def __init__(self, df : pd.DataFrame, fold_path : Path, seed : int, fold_num : int, logger) : 
        df = df[['save_image_path', 'patient_id', 'severity', f'FOLD_{fold_num}']].rename(columns = {'save_image_path' : 'image_path', f'FOLD_{fold_num}' : 'FOLD'})
        self.data_df = df
        self.fold_path = fold_path
        self.seed = seed
        self.fold_num = fold_num
        self.learning_logger = logger
        self.split_data()

        self.training_transform = transforms.Compose([        
                                                transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                # transforms.RandomRotation(90),
                                            ])
        self.test_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                            ]) 
    
    def split_data(self) : 
        if self.fold_path.joinpath(f'data_df_seed_{self.seed}.csv').exists() :
            self.data_df = pd.read_csv(self.fold_path.joinpath(f'data_df_seed_{self.seed}.csv'))
            return
        train_df = self.data_df.query(f'FOLD == 0')
        test_df = self.data_df.query(f'FOLD == 2')
        train_patient_df = train_df[['patient_id', 'severity']].drop_duplicates()
        train_patient_ids = train_patient_df['patient_id'].values
        train_labels = train_patient_df['severity'].values
        test_patient_ids = test_df['patient_id'].values
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.seed)
        for new_fold_num, (tr_idx, val_idx) in enumerate((skf.split(train_patient_ids, train_labels))) :
            tr_patient_ids = train_patient_ids[tr_idx]
            val_patient_ids = train_patient_ids[val_idx]
            self.data_df[f'FOLD_{new_fold_num}'] = self.data_df['patient_id'].apply(lambda x : 0 if x in tr_patient_ids else 1 if x in val_patient_ids else 2 if x in test_patient_ids else 3)
        self.data_df.to_csv(self.fold_path.joinpath(f'data_df_seed_{self.seed}.csv'), index = False)

    def load(self, fold_num: int, batch_size: int):
        
        def load_paths_and_labels(fold_num : int, label : int) : 
            image_paths = self.data_df[self.data_df[f'FOLD_{fold_num}'] == label ]['image_path'].values
            target_labels = self.data_df[self.data_df[f'FOLD_{fold_num}'] == label ]['severity'].values 
            return image_paths, target_labels

        train_paths, train_labels = load_paths_and_labels(fold_num, 0)
        valid_paths, valid_labels = load_paths_and_labels(fold_num, 1)
        test_paths , test_labels  = load_paths_and_labels(fold_num, 2)

        train_dset = CustomDataset(train_paths, train_labels, self.training_transform)
        valid_dset = CustomDataset(valid_paths, valid_labels, self.test_transform)
        test_dset  = CustomDataset(test_paths , test_labels , self.test_transform)

        dataloaders = {
            'TRAIN' : DataLoader(train_dset, batch_size = batch_size, shuffle = True, num_workers = 4),
            'VALID' : DataLoader(valid_dset, batch_size = batch_size, shuffle = False, num_workers = 4),
            'TEST'  : DataLoader(test_dset , batch_size = batch_size, shuffle = False, num_workers = 4)
        }
        return dataloaders















     