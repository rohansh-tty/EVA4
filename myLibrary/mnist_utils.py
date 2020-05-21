import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


# display 25 images with labels
def display_25(images, labels):
	figmatrix = plt.figure(figsize=(8, 8))
	row, col = 5, 5 # set rows & columns = 5
	gs = gridspec.GridSpec(row, col)
	gs.update(wspace=0.005, hspace=0.05)

	for i in range(1, 26):
		plt.subplot(gs[i-1])
		plt.tick_params( axis='both', which='both', labelbottom=False, labelleft=False, left=False, bottom=False)
		plt.imshow(images[i-1].numpy().squeeze(), cmap='gray_r')
		plt.text(2, 6, labels[i-1].numpy(), color="green", fontsize="xx-large")
	plt.show()


# display data
def mnist_data_display(num_of_images):
	for i in range(1,num_of_images+1):
		plt.subplot(6,10, i)
		plt.axis('off')
		plt.imshow(images[i-1].numpy().squeeze(), cmap = 'gray_r')


# Misclassified Images

from google.colab import files
def misclassifiedOnes(model, filename):
	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')	
	model = model.to(device)
	dataiter = iter(testloader) 
	count = 0
	fig = plt.figure(figsize=(13,13))

	while count<25:
		images, labels = dataiter.next()
		images, labels = images.to(device), labels.to(device)
    
		output = model(images) 
		_, pred = torch.max(output, 1)   # convert output probabilities to predicted class
		images = images.cpu().numpy() # conv images to numpy format

		for idx in np.arange(128):
			if pred[idx]!=labels[idx]:
				ax = fig.add_subplot(5, 5, count+1, xticks=[], yticks=[])
				count=count+1
				ax.imshow(np.squeeze(images[idx]), cmap='cool')
				ax.set_title("Pred-{} (Target-{})".format(str(pred[idx].item()), str(labels[idx].item())), color="Black")
				if count==25:
					break
		plt.savefig(filename)
	files.download(filename)

   








