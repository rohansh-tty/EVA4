import torch.nn as nn
from torchsummary import summary
import torch


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
	return summary(model, input_size)
