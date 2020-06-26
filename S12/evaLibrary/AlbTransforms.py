from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np

class AlbumentationTransforms:
  """
  Helper class to add Transformations to Test & Train Data using Albumentations
  """
  def __init__(self, transforms_list=[]):
    """
    Argument:
    transforms_list: List of Transformations from Albumentation Lib
    """
    
    transforms_list.append(AP.ToTensor()) # by default add ToTensor to conv the modified image to Tensor
    self.transforms = A.Compose(transforms_list)


  def __call__(self, img): # this is to make the instance callable
    img = np.array(img)
    return self.transforms(image=img)['image']

  