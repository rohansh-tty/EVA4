from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, VerticalFlip, Resize,Rotate #, Cutout
from albumentations.pytorch import ToTensor
import numpy as np


class test_transforms():

    def __init__(self, transforms_list=[
            Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
            ToTensor()]):
        self.albTestTransforms = Compose(transforms_list)
            
    def __call__(self, img):
        img = np.array(img)
        img = self.albTestTransforms(image=img)['image']
        return img
