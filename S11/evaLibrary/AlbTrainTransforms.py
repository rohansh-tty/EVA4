from albumentations import Compose, RandomCrop, Normalize, Resize, Rotate, Cutout, PadIfNeeded, RandomCrop, Flip
from albumentations.pytorch import ToTensor
import numpy as np


class train_transforms():
  """
  Transformations Applied:-

  1. PadIfNeeded: To pad the image by 4 pixels on each side, making the input image size = 40x40
  2. RandomCrop: To crop out 32x32 portion of the input image(40x400)
  3. Flip: Flip the input image horizontally or vertically at p = 0.50
  4. Cutout: Blocking a 8x8 portion of the input image at p = 1.0
  5. Normalize: Simply the normalizing the transformed image
  6. ToTensor: Converting the normalized image to Tensor
  """

    def __init__(self):
        self.albTrainTransforms = Compose([ 
            PadIfNeeded(min_height = 36, min_width = 36, border_mode = 0, p = 1.0),
            RandomCrop(height = 32, width = 32, p = 1.0),
            Flip(p=0.5),
            Cutout(num_holes = 1, max_h_size = 8, max_w_size = 8, p = 1.0),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albTrainTransforms(image=img)['image']
        return img
