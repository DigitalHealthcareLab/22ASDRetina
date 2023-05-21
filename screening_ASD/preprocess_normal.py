import sys
import os
root_path = os.path.abspath('..')
sys.path.append(root_path)
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from joblib import Parallel, delayed
import os
import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from src.seed import seed_everything


# 변경이 필요한 부분
## 경로설정
root_path = Path(root_path)
current_path = Path(os.getcwd())
num_folds = 10
seed = 2023
seed_everything(seed)
save_path = current_path.joinpath('processed_normal')
save_path.mkdir(exist_ok=True, parents=True)

model_name = 'ResNeXt50'
objective = 'TD'
df = pd.read_csv(f'age_sex_matched_df.csv')
total_path = Path(f'{model_name}_{objective}_seed_{seed}')
total_path.mkdir(parents=True, exist_ok=True)

info_df_path = root_path.joinpath('total_information.csv')
info_df = pd.read_csv(info_df_path)
info_df['raw_image_path'] = info_df['image_path'].apply(lambda x : root_path.joinpath(x.replace('ASD/', 'asd/').replace('TD/', 'td/')))
info_df['save_image_path'] = info_df['image_path'].apply(lambda x : current_path.joinpath(x.replace('data/', 'processed_normal/')))
image_path_raw_path_dict = dict(zip(info_df['image_path'], info_df['raw_image_path']))
image_path_save_path_dict =  dict(zip(info_df['image_path'], info_df['save_image_path']))

df['raw_image_path'] = df['image_path'].map(image_path_raw_path_dict)
df['save_image_path'] = df['image_path'].map(image_path_save_path_dict)

print(df.shape)


def load_image(image_path : Path) : 
    image = cv2.imread(image_path.as_posix())
    return image

def clahe_image(image : np.array) : 
    r, g, b = cv2.split(image)
    new_r = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(r)
    image = cv2.merge((new_r, g, b))
    return image

def resize_image(image : np.array, size : int) : 
    image = cv2.resize(image, (size, size))
    return image

def crop_top_bottom(image : np.array) : 
    # top bottom each 10% crop
    h, _, _ = image.shape
    image = image[int(h*0.1):int(h*0.9), :, :]
    return image

def process_single_iamge(image_path : Path, save_image_path : Path, size : int) : 
    try : 
        image = load_image(image_path)
        # image = clahe_image(image)
        image = crop_top_bottom(image)
        image = resize_image(image, size)
        save_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_image_path.as_posix(), image)
    except : 
        pass

Parallel(n_jobs=16)(delayed(process_single_iamge)(image_path, save_image_path, 224) for image_path, save_image_path in tqdm.tqdm(zip(df['raw_image_path'], df['save_image_path']), total = len(df)))



df['severity'] = df['data_class'].apply(lambda x : 1 if x =='ASD' else 0)

patient_df = df[['patient_id','severity']].drop_duplicates().reset_index(drop=True)
patient_ids = patient_df['patient_id'].values
labels = patient_df['severity'].values

train_ids, test_ids = train_test_split(patient_ids, test_size=0.15, random_state=seed, stratify=labels, shuffle=True)
df['FOLD'] = df['patient_id'].apply(lambda x : 0 if x in train_ids else 2 if x in test_ids else 999)

train_df = df.query('FOLD == 0')
train_patient_df = train_df[['patient_id','severity']].drop_duplicates().reset_index(drop=True)
train_patient_ids = train_patient_df['patient_id'].values
train_labels = train_patient_df['severity'].values

test_df = df.query('FOLD == 2')
test_patient_df = test_df[['patient_id','severity']].drop_duplicates().reset_index(drop=True)
test_patient_ids = test_patient_df['patient_id'].values
test_labels = test_patient_df['severity'].values

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

for fold_num, (train_index, val_index) in enumerate(skf.split(train_patient_ids, train_labels)):
    train_patient_id = train_patient_ids[train_index]
    val_patient_id = train_patient_ids[val_index]
    df[f'FOLD_{fold_num}'] = df['patient_id'].apply(lambda x : 0 if x in train_patient_id else 1 if x in val_patient_id else 2 if x in test_patient_ids else 999)

    print(f'FOLD_{fold_num} : {df.groupby([f"FOLD_{fold_num}", "severity"])["patient_id"].count()}')

df.to_csv(total_path.joinpath(f'preprocessed_df_{objective}.csv'), index=False)
