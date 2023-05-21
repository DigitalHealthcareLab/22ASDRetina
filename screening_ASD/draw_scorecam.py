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
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

# Modeling Libraries
import torch
import torch.nn as nn
import torch.optim
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,log_loss
from skimage.transform import resize

# Source Codes
from src.seed import seed_everything
from src.logger import print_logger
from src.model_ResNet import ResNeXt50
from src.gradcam_model import GradCamModel
from src.ScoreCAM.cam import ScoreCAM

objective = "TD"
num_folds=10
seed = 2023
seed_everything(seed)
device = 'cuda'

model_name = 'ResNeXt50'
model = ResNeXt50(2)

transform = transforms.Compose([
                        transforms.ToTensor(),
                    ]) 

total_path = Path(f'{model_name}_{objective}_seed_{seed}')
df_path = total_path.joinpath(f'preprocessed_df_TD.csv')
df = pd.read_csv(df_path)
df = df.query('FOLD == 2')

save_path = Path('ScoreCAM')
save_path.mkdir(exist_ok=True)

# ScoreCAM 세팅
model_dict = dict(type = 'ResNet', arch = model, layer_name = 'layer4', input_size = (224, 224))

transform = transforms.Compose([
                                transforms.ToTensor()
                            ])

def draw_single_image(model, outer_fold_num, inner_fold_num, image_path : Path) : 
    image_path = Path(image_path)
    data_class = image_path.parent.name
    image_stem = image_path.stem
    scorecam_save_path = save_path.joinpath(f"fold_{outer_fold_num}", f"fold_{inner_fold_num}", data_class, image_stem).with_suffix('.npy')
    scorecam_save_path.parent.mkdir(exist_ok=True, parents=True)
    if scorecam_save_path.exists() :
        return
    target_layer = model.layer4[2].conv3
    model_scorecam = ScoreCAM(model, target_layer, 8)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    cam, idx = model_scorecam(image)
    np.save(scorecam_save_path, cam)
    torch.cuda.empty_cache()

def process_single_fold(fold_num) : 
    outer_fold_num, inner_fold_num = fold_num // 5, fold_num % 5
    outer_fold_path = total_path.joinpath(f"fold_{outer_fold_num}")
    model_path = outer_fold_path.joinpath(f"model_{inner_fold_num}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    for image_path in tqdm(df['save_image_path'], total = len(df)) : 
        draw_single_image(model, outer_fold_num, inner_fold_num, image_path)
        


# Parallel(n_jobs=3)(delayed(process_single_fold)(fold_num) for fold_num in range(50))

for fold_num in [x for x in range(50)][::-1] : 
    process_single_fold(fold_num)