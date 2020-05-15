import torch.nn as nn
from torchsummary import summary


def cross_entropy_loss():
    """Create Cross Entropy Loss
    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()

def model_summary(model, input_size=(3,32,32)):
	"""
	Returns Summary of the model passed in as model
	"""
	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')
	print('Device set to ', device)
	return summary(model, input_size)
