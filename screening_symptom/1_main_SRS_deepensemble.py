from pathlib import Path
import sys
import numpy as np
import pandas as pd
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataloader_setter import Dataloader_Setter
from src.logger import print_logger
from src.earlystopping import EarlyStopping
from src.trainer import Trainer
from src.seed import seed_everything
from src.load_model import load_model
from src.data_splitting import split_dataset_undersampling

seed_everything(20230403)

################
prepared_images_path_rgb = '/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset'
prepared_images_path_rgb = Path(prepared_images_path_rgb)
ADOS_info_path = '/home/jaehan0605/MAIN_ASDfundus/DATA/Final_info.csv'
image_paths = np.array(list(prepared_images_path_rgb.rglob('*.png')))

file_name = [] ## File name
for i in range(len(image_paths)):
    for_append = []
    for_append.append(f'{image_paths[i].stem}')
    for_append.append(int(f'{image_paths[i].stem}'[2:6]))
    file_name.append(for_append)
file_name.sort()
file_name_df = pd.DataFrame(file_name, columns=['file_name', 'Number'])

ADOS_df = pd.read_csv(ADOS_info_path)
ADOS_df = ADOS_df.drop(ADOS_df[ADOS_df['age'] > 18].index)###################)###################)###################)###################)###################)###################)###################)###################

ADOS_DF = pd.merge(file_name_df, ADOS_df, on = 'Number', how='outer')
ADOS_DF = ADOS_DF.set_index('file_name')
ADOS_DF[f"cutoff_8_ADOS"] = ADOS_DF['ADOS'].apply(lambda x: 'severe' if x>=8 else ('non_severe' if i>x else ''))
ADOS_DF[f"cutoff_75_SRS"] = ADOS_DF['SRS_total'].apply(lambda x: 'severe' if x>75 else ('non_severe' if i>x else ''))

src = ['/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/severe/', '/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/non_severe/']
for src in src:
    dest = '/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/'
    files = os.listdir(src)
    for f in files:
        shutil.move(src + f, dest)

###라벨링에 따라서 raw image가 각 폴더로 이동
for a in ADOS_DF.index:
    if ADOS_DF.loc[a][f"cutoff_75_SRS"] == 'severe': 
        shutil.move(Path(f"/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/{a}.png"), '/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/severe')
    elif ADOS_DF.loc[a][f"cutoff_75_SRS"] == 'non_severe':
        shutil.move(Path(f"/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/{a}.png"), '/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/non_severe')
    else:
        pass

################
save_path = Path('/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset')
save_path1 = Path('/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/severe')
save_path2 = Path('/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset/non_severe')
save_path1.mkdir(exist_ok=True)
save_path2.mkdir(exist_ok=True)

split_dataset_undersampling(save_path = save_path, save_path1 = save_path1, save_path2 = save_path2)

################
total_path = Path('ResNeXt50_deepensemble_SRS_76')
num_epochs = 100
batch_size = 32
device = 'cuda'
initial_learning_rate = 1e-3

# LR Scheduler
scheduler_factor = 0.1
scheduler_patience = 3
scheduler_mode = 'min'
scheduler_min_lr = 0.0001

# Earlystops
earlystop_criterion = 'val_loss'
earlystop_patience = 10
earlystop_min_delta = 0.0001
earlystop_mode = 'minimize'

logger = print_logger(total_path, "Total_Logger.txt")
dataloader = Dataloader_Setter(logger)

if __name__ == "__main__" : 

        for fold_num in range(10) : 
            logger("=========================================")
            logger(f"Fold {fold_num} is started")

            fold_dataloader = dataloader.load(fold_num, batch_size)

            fold_path = total_path.joinpath(f'fold_{fold_num}')
            fold_logger = print_logger(fold_path, "training_log.txt")
            fold_logger(f"Fold {fold_num} start")
            fold_logger(f'Fold {fold_num}')
            fold_logger(f'TRAIN Dataloader : {fold_dataloader["TRAIN"].dataset.__len__()} | {np.bincount(fold_dataloader["TRAIN"].dataset.labels)}')
            fold_logger(f'VALID Dataloader : {fold_dataloader["VALID"].dataset.__len__()} | {np.bincount(fold_dataloader["VALID"].dataset.labels)}')
            fold_logger(f'TEST Dataloader : {fold_dataloader["TEST"].dataset.__len__()} | {np.bincount(fold_dataloader["TEST"].dataset.labels)}')        

            for model_num in range(5):
                globals()['model_{}'.format(model_num)] = load_model(model_name='ResNeXt50', pretrained=True, num_classes=2)
                model = globals()['model_{}'.format(model_num)].to(device)
                criterion = nn.CrossEntropyLoss().to(device)
                optimizer = Adam(model.parameters(), lr = initial_learning_rate, weight_decay=1e-6)
                scheduler = ReduceLROnPlateau(optimizer, factor = scheduler_factor, patience = scheduler_patience, mode = scheduler_mode, min_lr = scheduler_min_lr, verbose = True)
                early_stopping = EarlyStopping(earlystop_criterion, 
                                                patience=earlystop_patience, 
                                                verbose=True, 
                                                delta=earlystop_min_delta, 
                                                earlystop_mode = earlystop_mode, 
                                                path=fold_path.joinpath(f'Model_{model_num}.pth'), 
                                                trace_func=fold_logger)
                trainer = Trainer(model, fold_dataloader, criterion, optimizer, scheduler, fold_logger, device, early_stopping, initial_learning_rate)
                trainer.fit(num_epochs = num_epochs)
        

            logger(f"Fold {fold_num} is finished")
            logger('='*50)

