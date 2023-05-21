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
from tqdm import tqdm

# Modeling Libraries
import torch
import torch.nn as nn
import torch.optim
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,log_loss, roc_curve, auc

# Source Codes
from src.seed import seed_everything
from src.logger import print_logger
from src.model_ResNet import ResNeXt50
import matplotlib.pyplot as plt

seed_everything(20230403)
device = 'cuda'

model_name = 'ResNeXt50'
model = ResNeXt50(2)
transform = transforms.Compose([
                        transforms.ToTensor(),
                    ]) 

total_path = Path('ResNeXt50_TD_seed_2023')
df_path = total_path.joinpath(f'preprocessed_df_TD.csv')
df = pd.read_csv(df_path)
df = df.query('FOLD == 2')
df['image_path'] = df['save_image_path'].apply(lambda x : Path(x.split('/', 4)[-1]))
df['image_stem'] = df['image_path'].apply(lambda x : x.stem)
df['severity'] = df['data_class'].apply(lambda x : 1 if x == 'ASD' else 0)

gradcam_path = Path('ScoreCAM')
save_path = Path('PEPPR_ScoreCAM')

def process_inner_fold(fold_num) : 
    outer_fold_num, inner_fold_num = fold_num // 5, fold_num % 5
    print('process outer fold : ', outer_fold_num, 'process inner fold : ', inner_fold_num)
    outer_fold_name = f'fold_{outer_fold_num}'
    outer_fold_path = total_path.joinpath(outer_fold_name)
    outer_fold_save_path = save_path.joinpath(outer_fold_name)
    gradcam_outer_fold_path = gradcam_path.joinpath(outer_fold_name)
    test_df = df[['image_path', 'severity', 'image_stem']]

    model_path = outer_fold_path.joinpath(f"model_{inner_fold_num}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    gradcam_inner_fold_path = gradcam_path.joinpath(model_path.parent.name, f"fold_{inner_fold_num}")
    inner_test_df = test_df.copy()
    inner_fold_save_path = outer_fold_save_path.joinpath(f"fold_{inner_fold_num}")
    
    results = []
    with torch.no_grad() : 
        for image_path, label in zip(inner_test_df['image_path'], inner_test_df['severity']) : 
            label_name = 'ASD' if label == 1 else 'TD'
            # print(label, label_name, image_path)
            image_stem = image_path.stem
            image = cv2.imread(image_path.as_posix(), cv2.COLOR_BGR2RGB)
            label = torch.tensor(label).unsqueeze(0).to(device)
            gradcam_array_path = gradcam_inner_fold_path.joinpath(label_name, image_path.stem).with_suffix('.npy')

            gradcam = np.load(gradcam_array_path)
            if gradcam.ndim == 4 : 
                gradcam = gradcam[0][0]

            
            image_result = pd.DataFrame(columns = ['image_path'] + [f'quantile_{quantile:.2f}' for quantile in np.arange(0, 1.01, 0.05)])
            image_result['image_path'] = [image_path]
            for quantile in np.arange(0, 1.01, 0.05) : 
                if quantile > 0 : 
                    quantile_value = np.quantile(gradcam, quantile)
                    gradcam_mask = np.where(gradcam > quantile_value, 1, 0)
                    new_image = image * np.stack([gradcam_mask, gradcam_mask, gradcam_mask], axis = 2)
                    new_image = np.array(new_image, dtype = np.uint8)
                else :
                    new_image = image
                image_save_path = inner_fold_save_path.joinpath(f'quantile_{quantile:.2f}',image_path.name)
                if image_save_path.exists() : 
                    continue
                image_tensor = transform(new_image).unsqueeze(0).to(device)
                output = torch.softmax(model(image_tensor), dim = 1)[0][1].item()
                
                image_save_path.parent.mkdir(parents = True, exist_ok = True)
                cv2.imwrite(image_save_path.as_posix(), new_image)
                image_result[f'quantile_{quantile:.2f}'] = [output]
            results.append(image_result)
        
    result = pd.concat(results).reset_index(drop = True)
    result.to_csv(inner_fold_save_path.joinpath('PEPPR_inference_result.csv'), index = False)
 
Parallel(n_jobs = 10)(delayed(process_inner_fold)(fold_num) for fold_num in range(50)) ##########################여기 50으로 바꿔야함

def make_deep_ensemble(fold) : 
    fold_path = gradcam_path.joinpath(f"fold_{fold}")
    save_path = fold_path.joinpath("deep_ensemble")
    save_path.mkdir(parents = True, exist_ok = True)
    array_paths = [Path(array_path.parent.name).joinpath(array_path.name) for array_path in fold_path.joinpath('fold_0').rglob('*.npy')]
    for array_path in tqdm(array_paths, total = len(array_paths)) : 
        inner_fold_arrays = []
        for inner_fold_num in range(5) : 
            inner_fold_path = fold_path.joinpath(f"fold_{inner_fold_num}").joinpath(array_path)
            inner_fold_array = np.load(inner_fold_path)[0][0]
            inner_fold_arrays.append(inner_fold_array)
        inner_fold_arrays = np.array(inner_fold_arrays).mean(axis = 0)
        np.save(save_path.joinpath(array_path.name), inner_fold_arrays)
    
for fold_num in range(10) : 
    make_deep_ensemble(fold_num)