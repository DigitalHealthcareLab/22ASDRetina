from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import json
from joblib import Parallel, delayed
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import tqdm
from PIL import Image, ImageDraw, ImageOps

# non_severe_data_path = Path('/home/jaehan0605/MAIN_ASDfundus/DATA/raw_image/non_severe')
# severe_data_path = Path('/home/jaehan0605/MAIN_ASDfundus/DATA/raw_image/severe')

# save_path = Path('/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset')
# save_path.mkdir(exist_ok=True)

non_severe_data_path = Path('/home/jaehan0605/MAIN_ASDfundus/OOD_images/non_severe')
severe_data_path = Path('/home/jaehan0605/MAIN_ASDfundus/OOD_images/severe')

save_path = Path('/home/jaehan0605/MAIN_ASDfundus/OOD_images/Processed_dataset')
save_path.mkdir(exist_ok=True)

def load_image_paths() : 

    def __init__(self):
        return

    def load_non_severe_image_paths() :
        non_severe_image_paths = np.array(list(non_severe_data_path.rglob('*.png')) + list(non_severe_data_path.rglob('*.jpg')) + list(non_severe_data_path.rglob('*.jpeg')) + list(non_severe_data_path.rglob('*.JPG')))
        non_severe_image_paths = [[x, 'non_severe'] for x in non_severe_image_paths]
        return non_severe_image_paths

    def load_severe_image_paths() :
        severe_image_paths = np.array(list(severe_data_path.rglob('*.png')) + list(severe_data_path.rglob('*.jpg')) + list(severe_data_path.rglob('*.jpeg')) + list(severe_data_path.rglob('*.JPG')))
        severe_image_paths = [[x, 'severe'] for x in severe_image_paths]
        return severe_image_paths


    non_severe_image_paths = load_non_severe_image_paths()
    severe_image_paths = load_severe_image_paths()

    print(f'non_severe image counts : {len(non_severe_image_paths)}') 
    print(f'severe image counts : {len(severe_image_paths)}')   

    total_image_paths = non_severe_image_paths + severe_image_paths

    image_path_df = pd.DataFrame(
        total_image_paths,
        columns = ['image_path', 'data_class']
    )

    return image_path_df

image_path_df = load_image_paths()

image_paths = image_path_df['image_path'].apply(lambda x: Path(x)).values

# processed_save_path = Path('/home/jaehan0605/MAIN_ASDfundus/DATA/Processed_dataset')
processed_save_path = Path('/home/jaehan0605/MAIN_ASDfundus/OOD_images/Processed_dataset')

image_path_df['processed_save_path'] = image_path_df['image_path'].apply(lambda x : processed_save_path.joinpath('/'.join(Path(x).parts[6:])))
save_paths = image_path_df['processed_save_path'].values

def remove_black_area_1(image_path : Path) -> np.array: 

    if image_path.stem in ['1_0058_M_04_ASD_L', '1_0012_M_06_ASD_L', '1_0362_F_04_ASD_R', '1_0294_M_17_ASD_R']:
        tol = 50
    else:
        tol = 10

    color_image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    gray_image = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE) # mask 만드려면 지우면 안댐!
    mask = gray_image > tol

    return color_image[np.ix_(mask.any(1), mask.any(0))]


def crop_circle(image : np.array) -> None : 
    height, width, channel = image.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x,y))
    circle_image = np.zeros((height, width), np.uint8)
 
    cv2.circle(circle_image, (x,y), int(r), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_image)    ### cv2.bitwise_and(이미지파일1, 이미지파일2, 적용영역지정)
    return image


def remove_black_area_2(image: np.array):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # mask 만드려면 지우면 안댐!
    tol=0
    mask = gray_image > tol
    return image[np.ix_(mask.any(1), mask.any(0))]

from torchvision import transforms
import torch

def process_single_image(image_path : Path, image_save_path : Path) : 

    image = remove_black_area_1(image_path)
    image = crop_circle(image)
    image = remove_black_area_2(image)
    image = cv2.resize(image, (224, 224))

    image_save_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(image_save_path.as_posix(), image)
    return True

for i, j in tqdm.tqdm(zip(image_paths, save_paths)):
    process_single_image(i, j)

