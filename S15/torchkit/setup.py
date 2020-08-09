import os
import torch
import zipfile
import time as t
import torchvision
from tqdm import notebook
import torch.nn.functional as F
from IPython.display import Image, clear_output 
from tensorboardcolab import TensorBoardColab
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD, Adam



# Hardware Properties
def hardware_specs():
    return 'PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU')

