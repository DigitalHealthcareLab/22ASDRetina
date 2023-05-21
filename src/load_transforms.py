from torchvision import transforms
import numpy as np
import cv2
import torch

class preprocess_transform(object) : 
    def __init__(self, preprocess_type) : 
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.preprocess_type = preprocess_type

    def clahe_image(self, image : np.array) : 

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab = cv2.merge((l_clahe, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image


    def __call__(self, image : np.array) : 
        if self.preprocess_type == 'clahe' :
            return self.clahe_image(image)
        else : 
            return image



class channel_transform(object) : 
    def __init__(self, image_channel : str) : 
        self.image_channel = image_channel
    
    def rgb_image(self, image : np.array) : 
        return image
    
    def r_image(self, image : np.array) :
        r, g, b = cv2.split(image)
        return np.stack([r, r, r], axis=-1)
    
    def g_image(self, image : np.array) :
        r, g, b = cv2.split(image)
        return np.stack([g, g, g], axis=-1)
    
    def b_image(self, image : np.array) :
        r, g, b = cv2.split(image)
        return np.stack([b, b, b], axis=-1)
    
    def gb_image(self, image : np.array) :
        image[:, :, 0] = 0
        return image
    
    def gray_image(self, image : np.array) : 
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.stack([gray, gray, gray], axis=-1)
    
    def redfree_gray_image(self, image : np.array) :
        r, g, b = cv2.split(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = np.stack([gray, gray, gray], axis=-1)
        gray[:, :, 0] = r
        return gray
    
    def gb_gray_image(self, image : np.array) :
        image = self.gb_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = np.stack([gray, gray, gray], axis=-1)
        return gray
    
    def __call__(self, image : np.array) : 
        if self.image_channel == 'rgb' : 
            return self.rgb_image(image)
        elif self.image_channel == 'r' :
            return self.r_image(image)
        elif self.image_channel == 'g' :
            return self.g_image(image)
        elif self.image_channel == 'b' :
            return self.b_image(image)
        elif self.image_channel == 'gb' :
            return self.gb_image(image)
        elif self.image_channel == 'gray' :
            return self.gray_image(image)
        elif self.image_channel == 'redfree_gray' :
            return self.redfree_gray_image(image)
        elif self.image_channel == 'gb_gray' :
            return self.gb_gray_image(image)


def load_transforms(phase : str, preprocess_type : str, image_channel : str) : 


    transform = [preprocess_transform(preprocess_type), channel_transform(image_channel)]

    if phase == 'train' : 
        transform = transform + [
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()
                                    ]
    elif phase == 'valid' or phase == 'test' :
        transform = transform + [
                                    transforms.ToTensor()
                                    ]
    transform = transforms.Compose(transform)
    return transform