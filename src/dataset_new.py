from pathlib import Path
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms

class customDataset(Dataset):
    def __init__(self, csv_file, mode, transform = None) : 
        df = pd.read_csv(csv_file).query('split == @mode')
        image_paths = df['image_path'].values
        self.images = np.stack([cv2.imread(image_path) for image_path in image_paths])
        if transform is None : 
            if mode == 'train' : 
                self.mean = self.images.mean(axis=(0, 1, 2)) / 255
                self.std = self.images.std(axis=(0, 1, 2)) / 255
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(0.4), 
                    transforms.RandomVerticalFlip(0.1),
                    transforms.Normalize(self.mean, self.std)
                ])
            else :
                assert mode == 'valid', 'transform must be defined if mode is not train'
        else : 
            self.transform = transform
            
    
    def __len__(self) : 
        return len(self.images)
    
    def __getitem__(self, idx) :
        image = self.images[idx]
        image = Image.fromarray(image)
        image = self.transform(image)
        
        random_idx = np.random.randint(0, len(self.images))
        random_image = self.images[random_idx]
        random_image = Image.fromarray(random_image)
        random_image = self.transform(random_image)
        return image, random_image
    
    def get_mean_std(self) : 
        return self.mean, self.std

  # train_dataset = customDataset(metadata_filename, 'train')
  # transforms = train_dataset.transform
  # valid_dataset = customDataset(metadata_filename, 'valid', transform=transforms)
