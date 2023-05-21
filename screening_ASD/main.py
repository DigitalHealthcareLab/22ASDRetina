import sys
import os
root_path = os.path.abspath('..')
sys.path.append(root_path)

# Basic Libraries
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from datetime import datetime

# Modeling Libraries
import torch
import torch.nn as nn
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Source Codes
from src.seed import seed_everything
from src.logger import print_logger
from src.earlystopping import EarlyStopping
from src.dataloader_setter import Dataloader_Setter
from src.model_ResNet import ResNeXt50
from src.trainer import Trainer

objective = "TD"
num_folds = 10
seed = 2023
seed_everything(seed)

model_name = 'ResNeXt50'
total_path = Path(f'{model_name}_{objective}_seed_{seed}')

df_path = total_path.joinpath(f'preprocessed_df_{objective}.csv')
df = pd.read_csv(df_path)
print(df)


### Modeing Configures
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

for outer_fold_num in range(num_folds) : 
    fold_path = total_path.joinpath(f"fold_{outer_fold_num}")
    fold_path.mkdir(parents = True, exist_ok = True)
    dataloader_setter = Dataloader_Setter(df, fold_path, seed, outer_fold_num, logger)
    
    fold_logger = print_logger(fold_path, "training_log.txt")    
    for inner_fold_num in range(5) : 
        fold_dataloader = dataloader_setter.load(inner_fold_num, batch_size)
        model_save_path = fold_path.joinpath(f"model_{inner_fold_num}.pth")
        fold_logger(f"Fold {inner_fold_num} start")
        fold_logger(f'Fold {inner_fold_num}')
        fold_logger(f'TRAIN Dataloader : {fold_dataloader["TRAIN"].dataset.__len__()} | {np.bincount(fold_dataloader["TRAIN"].dataset.labels)}')
        fold_logger(f'VALID Dataloader : {fold_dataloader["VALID"].dataset.__len__()} | {np.bincount(fold_dataloader["VALID"].dataset.labels)}')
        fold_logger(f'TEST Dataloader : {fold_dataloader["TEST"].dataset.__len__()} | {np.bincount(fold_dataloader["TEST"].dataset.labels)}')        
        model = ResNeXt50(num_classes = 2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr = initial_learning_rate, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, 
                                        factor = scheduler_factor, 
                                        patience = scheduler_patience, 
                                        mode = scheduler_mode, 
                                        min_lr = scheduler_min_lr)
        early_stopping = EarlyStopping(earlystop_criterion = earlystop_criterion, 
                                        patience = earlystop_patience, 
                                        delta = earlystop_min_delta, 
                                        earlystop_mode = earlystop_mode,
                                        path = model_save_path)
        trainer = Trainer(model = model,
                            dataloaders = fold_dataloader,
                            criterion = criterion,
                            optimizer = optimizer,
                            scheduler = scheduler,
                            logger = fold_logger,
                            device = device,
                            earlystop = early_stopping,
                            initial_learning_rate = initial_learning_rate
                            )
        trainer.fit(num_epochs)

    fold_logger(f"Fold {inner_fold_num} end")
    fold_logger(f"=" * 50)
        