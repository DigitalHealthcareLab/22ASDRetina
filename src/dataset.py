from pathlib import Path
import numpy as np
import cv2
from torch.utils.data.dataset import TensorDataset


class CustomDataset(TensorDataset) : 

    def __init__(self, image_paths : list, labels, transform) : 
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform  
        
    def __getitem__(self, idx) : 
        path = self.image_paths[idx]
        path = Path(path)
        data = cv2.imread(path.as_posix(), cv2.COLOR_BGR2RGB)
        data = self.transform(data)
        label = self.labels[idx]
        return data, label, path.stem


    def __len__(self) : 
        return len(self.image_paths)