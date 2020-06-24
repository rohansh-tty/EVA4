import torch

class DataLoaders:
  def __init__(self, 
              batch_size=512,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              seed=1):
    """
    Arguments:-
    batch_size: Number of images to be passed in each batch
    shuffle(boolean):  If True, then shuffling of the dataset takes place
    num_workers(int): Number of processes that generate batches in parallel
    pin_memory(boolean):
    seed: Random Number, this is to maintain the consistency
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')  # set device to cuda

    if use_cuda:
      torch.manual_seed(seed)
    
    self.dataLoader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True ) if use_cuda else dict(batch_size=1, shuffle=True, num_workers = 1, pin_memory = True)


  def dataLoader(self, data):
    return torch.utils.data.DataLoader(data,**self.dataLoader_args)
