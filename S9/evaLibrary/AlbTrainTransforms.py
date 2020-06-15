from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, VerticalFlip, Resize, Rotate,  Cutout, Flip
from albumentations.pytorch import ToTensor
import numpy as np


class train_transforms():

    def __init__(self):
        self.albTrainTransforms = Compose([  # Resize(256, 256),
            Rotate((-10.0, 10.0)),
            Flip(p=0.5),
            Cutout(num_holes = 1, max_h_size = 8, max_w_size = 8, p = 1.0),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensor()
        ])

    print("REQUIRED LIBRARIES LOADED...")

    def __call__(self, img):
        img = np.array(img)
        img = self.albTrainTransforms(image=img)['image']
        return img