from torchvision import transforms
import numpy as np
import random
from albumentations import HorizontalFlip, Compose, RandomCrop, Normalize, Rotate, pytorch
from albumentations.pytorch import ToTensor

# create a albumentation class to define test & train transformations
class albTransforms:
  def __init__(self, train = True):
    transformsList = []
    channel_means = (0.5, 0.5, 0.5)
    channel_stdevs = (0.5, 0.5, 0.5)

    if train:
      transformsList += [Rotate(-10.0, 10,0)]
      transformsList += [HorizontalFlip(0.5)]
      # transformsList += [RandomCrop(height = 2, width = 2, p=0.5)]

    transformsList += [Normalize(mean = channel_means, std = channel_stdevs, always_apply=True),
                       ToTensor()]

    self.transform = Compose(transformsList)


  def __call__(self, image):
    """Process the image through data transformation pipeline """

    img = np.array(image) # conv image to numpy
    img = self.transform(image = img)['image'] # transformation
    return img
