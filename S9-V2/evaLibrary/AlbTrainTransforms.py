from albumentations import Compose, RandomCrop, Normalize, Resize, Rotate, Cutout, PadIfNeeded, RandomCrop, Flip
from albumentations.pytorch import ToTensor
import numpy as np


class train_transforms():
    
    def __init__(self):
        self.albTrainTransforms = Compose([ 
            PadIfNeeded(min_height = 36, min_width = 36, border_mode = 0, p = 1.0),
            RandomCrop(height = 32, width = 32, p = 1.0),
            Rotate((-15.0,15.0), p=0.7),
            Cutout(num_holes = 2, max_h_size = 8, max_w_size = 8, p = 1.0),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albTrainTransforms(image=img)['image']
        return img