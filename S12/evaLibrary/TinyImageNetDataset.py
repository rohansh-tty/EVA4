import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image 
import csv

class tinyImageDataset:

    def __init__(self, 
                path, 
                download=True, 
                splitRatio=0.70, 
                random_seed=110, 
                transform=None):
        self.path=path
        self.download=download
        self.split=split
        self.random_seed=random_seed
        self.transform=transform

        # download the dataset
        if download:
            self.downloadDataset()
        
        if splitRatio > 1:
            raise ValueError("train_split must be less than 1")

        self._classID = self._mapID_to_className()  
        self.data, self.target = self._loadData()

        # shuffle the dataset
        self._imageIndex = np.arange(self.target)
        np.random.seed(random_seed)
        np.random.shuffle(self._imageIndex)

        # Split the data using Image indices
        splitValue = int(len(self._imageIndex)*self.splitRatio) 
        self._imageIndex = self._imageIndex[:splitValue]  if train else self._imageIndex[splitValue:]


    def __len__(self):
        return len(self._imageIndex) 


    def __getitem__(self, index):
        image_index = self._imageIndex[index]
        image = self.data[image_index]   

        if self.transform:
            image = self.transform(image)

        return image, self.target[image_index]

    
    def __repr__(self):
        head = 'TinyImageNet Dataset'
        body = ['Number of datapoints: {}'.format(self.__len__())]
        if self.path is not None:
            body.append('Root Location: {}'.format(self.path))
        body += [f'Split: {"Train" if self.train else "Test"}']
        if hasattr(self, 'transforms') and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)


    def tinyClasses(self):
        return tuple(cls[1]['name']  for cls in sorted(self._classID, key=lambda x:x[1]['id']))     

    def _mapID_to_className(self):
        with open(os.path.join(self.path, 'wnids.txt')) as f:
            class_ids = {x[:-1]: '' for x in f.readlines()} 

        with open(os.path.join(self.path, 'words.txt')) as f:
            class_id = 0
            for line in csv.reader(f, delimiter='\t'):
                if line[0] in class_ids:
                    class_ids[line[0]] = {"name": line[1], "id": class_id}
                    class_id += 1
        
        return class_ids 

    def _loadImage(self, image_path):
        img = Image(image_path)

        if img.mode == 'L':
            img_array = np.array(img) #convert image to np arra
            img = np.stack((img_array,)*3, axis=-1)
            img = Image.fromarray(img.astype('uint8'), 'RGB') #converting the np image back normal

        return img
    
    def _loadData(self):
        data = []
        target = []

        train_path = os.path.join(self.path, "train")
        for classDir in os.listdir(train_path):
            trainData_path = os.path.join(train_path, classDir)
            for image in os.listdir(trainData_path):
                if image.lower().endswith(".jpeg"):
                    data.append(self._loadImage(trainData_path, image))
                    target.append(self._classID[classDir]["id"])
        

        
        val_path = os.path.join(self.path, 'val')
        valImages_path = os.path.join(val_path, 'images')
        with open(os.path.join(val_path, 'val_annotations.txt')) as f:
            for line in csv.reader(f, delimiter='\t'):
                data.append(self._loadImage(os.path.join(valImages_path, line[0])))
                target.append(self._classID[line[1]]['id'])
        
        return data, target
    
    def downloadDataset(self):
        if not os.path.exists(self.path):
            r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
            zip_ref = zipfile.ZipFile(BytesIO(r.content))
            zip_ref.extractall(os.path.dirname(self.path))
            zip_ref.close()

            # Move file to appropriate location
            os.rename(os.path.join(os.path.dirname(self.path), 'tiny-imagenet-200'), self.path)
        else:
            print('Files already downloaded.')

def loader(self, train=True):
        """Create data loader.
        Args:
            train (bool, optional): True for training data. (default: True)
        
        Returns:
            Dataloader instance.
        """

        loader_args = {
            'batch_size': 128
            'num_workers': 4,
            'pin_memory': True
        }

        return data_loader(
            self.train_data, **loader_args
        ) if train else data_loader(self.val_data, **loader_args)


def data_loader(data, shuffle=True, batch_size=1, num_workers=1, cuda=False):
    """Create data loader
    Args:
        data (torchvision.datasets): Downloaded dataset.
        shuffle (bool, optional): If True, shuffle the dataset. 
            (default: True)
        batch_size (int, optional): Number of images to considered
            in each batch. (default: 1)
        num_workers (int, optional): How many subprocesses to use
            for data loading. (default: 1)
        cuda (bool, optional): True is GPU is available. (default: False)
    
    Returns:
        DataLoader instance.
    """

    loader_args = {
        'shuffle': shuffle,
        'batch_size': batch_size
    }

    # If GPU exists
    if cuda:
        loader_args['num_workers'] = num_workers
        loader_args['pin_memory'] = True
    
    return torch.utils.data.DataLoader(data, **loader_args)
