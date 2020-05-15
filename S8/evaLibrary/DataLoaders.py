import torch

class DataLoaders:
  def __init__(self, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, seed=1):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')  # set device to cuda

    if use_cuda:
      torch.manual_seed(seed)
    
    self.dataLoader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True ) if use_cuda else dict(batch_size=batch_size/2, shuffle=True)

  def dataLoader(self, data):
    return torch.utils.data.DataLoader(data,**self.dataLoader_args )


