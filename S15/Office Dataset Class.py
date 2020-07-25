from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile

def zip_data(zipfile, path_to_zipfile, directory_to_extract):
  with zipfile.Zipfile(path_to_zipfile, 'r') as zip:
    zip.extractall(directory_to_extract)


class OfficeDataset(Dataset):
  """
  Dataset Helper Class with simple and easy functions.

  Arguments: 
  data:  ZIP File consisting of Background Image(BG), Foreground overlayed on Background(FG_BG), Mask of FG_BG, Depth Map of FG_BG
  transforms: Tuple consisting of Image Transformations for all 4 Image types

  Follow this order BG, FG_BG, DEPTH MAP, MASK.

  [NOTE]: FG_BG and BG_FG are same
  """
  def __init__(self, data, transforms=(None,None,None,None)):
    self.bg_images, self.bgfg_images, self.mask_images, self.depth_maps = zip(*data) # do the unzipping part
    self.bg_transforms = transforms[0]
    self.bgfg_transforms = transforms[1]
    self.depthmap_transforms = transforms[2]
    self.mask_transforms = transforms[3]


  def __len__(self):
    """
    returns len of the dataset
    """
    return len(self.bgfg_images)


  def __getitem__(self, idx):
    """
    returns image data & target for the corresponding index
    """
    bg_image = Image.open(self.bg_images[idx])
    bgfg_image = Image.open(self.bgfg_images[idx])
    depth_map = Image.open(self.depth_maps[idx])
    mask_image = Image.open(self.mask_images[idx])


    # bg_image = self.bg_transforms(bg_image)
    # bgfg_image = self.bgfg_transforms(bgfg_image)
    # depth_map = self.depthmap_transforms(depth_map)
    # mask_image = self.mask_transforms(mask_image)


    return {"bg": bg_image, "bgfg": bgfg_image, "depthmap": depth_map, "mask": mask_image}
