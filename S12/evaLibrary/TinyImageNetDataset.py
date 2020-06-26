from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook



def TinyImageNetDataSet(splitRatio = 70, test_transforms = None, train_transforms = None):
  classes = extract_classnames(path = "tiny-imagenet-200/wnids.txt")
  data = TinyImageNet(classes, defPath="tiny-imagenet-200")
  traindata_len = len(data)*splitRatio//100
  testdata_len = len(data) - traindata_len 
  train, val = random_split(data, [traindata_len, testdata_len])
  train_dataset = DatasetFromSubset(train, transform=train_transforms)
  test_dataset = DatasetFromSubset(val, transform=test_transforms)

  return train_dataset, test_dataset, classes


# Extract class names out of wninds.txt
def extract_classnames(path = "tiny-imagenet-200/wnids.txt"):
  IDFile = open(path, "r")
  classes = []
  for line in IDFile:
    classes.append(line.strip())
  return classes



# TinyImageNet class
class TinyImageNet(Dataset):
    def __init__(self, classes, defPath):
        """
        Custom Dataset Class with couple of helper funcs

        Arguments:
        classes: Dataset Classes
        defPath: Common Path for Dataset Files
        """
        
        self.classes = classes
        self.defPath = defPath
        self.data = []
        self.target = []
        
        # Open Class ID File in read mode
        wnids = open(f"{defPath}/wnids.txt", "r")


        # Training Part
        trainImagePath = defPath+"/train/"
        for cls in notebook.tqdm(wnids, total = 200):
          cls = cls.strip() # strip spaces out of class names
          indFolderPath = trainImagePath + cls + "/images/"
          for i in os.listdir(indFolderPath):
            img = Image.open(indFolderPath + i)
            npimage = np.asarray(img)
            
            if(len(npimage.shape) == 2):
              npimage = np.repeat(npimage[:, :, np.newaxis], 3, axis=2) # add a new dim using np.newaxis, if it's a 2D
            
            self.data.append(npimage)  
            self.target.append(self.classes.index(cls))

        # Validation Part
        valAnntns = open(f"{defPath}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(valAnntns, total =10000):
          img, cls = i.strip().split("\t")[:2]
          img = Image.open(f"{defPath}/val/images/{img}")
          npimage = np.asarray(img)
          
          if(len(npimage.shape) == 2):  
                npimage = np.repeat(npimage[:, :, np.newaxis], 3, axis=2)
          
          self.data.append(npimage)  
          self.target.append(self.classes.index(cls))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data     
        return data,target



class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        """
        DatasetFromSubset: Class for loading split data and transforming the same

        Arguments:
        subset: Split part of the Dataset
        transform: List of transforms that needs to applied, by default None
        """
        self.subset = subset
        self.transform = transform


    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

