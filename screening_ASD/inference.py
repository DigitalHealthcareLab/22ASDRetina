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
import cv2
from joblib import Parallel, delayed
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

# Modeling Libraries
import torch
import torch.nn as nn
import torch.optim
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,log_loss

# Source Codes
from src.seed import seed_everything
from src.logger import print_logger
from src.model_ResNet import ResNeXt50

objective = "TD"
num_folds = 10
seed = 2023
seed_everything(seed)
device = 'cuda'

model_name = 'ResNeXt50'
model = ResNeXt50(2)

transform = transforms.Compose([
                        transforms.ToTensor(),
                    ]) 

total_path = Path(f'{model_name}_{objective}_seed_{seed}')
df_path = total_path.joinpath(f'preprocessed_df_{objective}.csv')
df = pd.read_csv(df_path)
df = df.query('FOLD == 2')


def process_inner_fold(fold_num) : 
    model_path = outer_fold_path.joinpath(f"model_{fold_num}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    inner_fold_pred_dict = {}
    with torch.no_grad() : 
        for image_path, label in zip(test_df['image_path'], test_df['severity']) : 
            image = cv2.imread(image_path.as_posix(), cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(device)
            label = torch.tensor(label).to(device)
            pred = torch.softmax(model(image), dim = 1)[0][1].item()
            inner_fold_pred_dict[image_path] = pred
    return inner_fold_pred_dict

def calculate_scores(labels, soft_preds) : 
    hard_preds = np.where(soft_preds > 0.5, 1, 0)

    auc_score   = roc_auc_score(labels, soft_preds)
    sensitivity = recall_score(labels, hard_preds)
    specificity = recall_score(labels, hard_preds, pos_label=0)
    precision   = precision_score(labels, hard_preds)
    f1 = f1_score(labels, hard_preds)
    accuracy    = accuracy_score(labels, hard_preds)
    nll         = log_loss(labels, hard_preds, eps=1e-15, normalize=True)
    brier       = brier_score_loss(labels, hard_preds)
    print(f"""
    AUC : {auc_score:.20f}
    Sensitivity : {sensitivity:.20f}
    Specificity : {specificity:.20f} 
    Precision : {precision:.20f} 
    F1 : {f1:.20f} 
    Accuracy : {accuracy:.20f} 
    NLL : {nll:.20f} 
    Brier : {brier:.20f}
    """)
    return auc_score, sensitivity, specificity, precision, f1, accuracy, nll, brier
    



total_pred_dict = {}
for outer_fold_num in range(10) : 
    print(f"Outer Fold : {outer_fold_num}")
    outer_fold_path = total_path.joinpath(f"fold_{outer_fold_num}")
    outer_fold_result_save_path = outer_fold_path.joinpath(f"inference_result.csv")
    if outer_fold_result_save_path.exists() :
        test_df = pd.read_csv(outer_fold_result_save_path)
        image_stem_pred_dict = zip(test_df['image_stem'], test_df['deep_ensemble'])
        total_pred_dict.update(image_stem_pred_dict)
        labels = test_df['severity'].values
        preds = test_df['deep_ensemble'].values
        calculate_scores(labels, preds)
        continue
    test_df = df[['save_image_path', 'severity']].rename(columns = {'save_image_path' : 'image_path'})
    test_df['image_path'] = test_df['image_path'].apply(lambda x : Path(x))
    test_df['image_stem'] = test_df['image_path'].apply(lambda x : x.stem)

    inner_fold_pred_dicts = Parallel(n_jobs = 1)(delayed(process_inner_fold)(inner_fold) for inner_fold in range(5))
    for inner_fold_num, inner_fold_pred_dict in enumerate(inner_fold_pred_dicts) : 
        test_df[f'pred_{inner_fold_num}'] = test_df['image_path'].map(inner_fold_pred_dict)
    test_df['deep_ensemble'] = test_df.filter(like='pred_').mean(axis=1)
    test_df.to_csv(outer_fold_result_save_path, index = False)
    image_stem_pred_dict = zip(test_df['image_stem'], test_df['deep_ensemble'])
    total_pred_dict.update(image_stem_pred_dict)

    labels = test_df['severity'].values
    preds = test_df['deep_ensemble'].values
    print(test_df)
    np.save(outer_fold_path.joinpath("patient_level_labels.npy"), labels)
    np.save(outer_fold_path.joinpath("patient_level_preds.npy"), preds)
    calculate_scores(labels, preds)
    
total_df = df[['save_image_path', 'severity']].rename(columns = {'save_image_path' : 'image_path'})
total_df['image_stem'] = total_df['image_path'].apply(lambda x : Path(x).stem)
total_df['pred'] = total_df['image_stem'].map(total_pred_dict)
total_df.to_csv(total_path.joinpath(f"total_inference_result.csv"), index = False)