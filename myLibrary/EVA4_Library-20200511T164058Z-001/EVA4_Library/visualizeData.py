#Visualization Functions
import matplotlib.pyplot as plt
import numpy as np
import torch


# Display Images from training dataset
def plotImage(img):
    img = img / 2 + 0.5  # unnormalize this is make sure the image is visible, if this step is skipped then the resulting images have a dark portion
    npimg = img.numpy()   # converting image to numpy array format
    plt.imshow(np.transpose(npimg, (1, 2, 0)))    # transposing npimg array




# Misclassified Images
from google.colab import files
def misclassifiedOnes(model, testloader, data,filename):

  #model: ModelName
  #data: Incorrect Classes in Test() of Test_Train class
  #filename: Pass on the filename with which you want to save misclassified images
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = model.to(device)
  dataiter = iter(testloader) 
  count = 0
  
  # Initialize plot
  fig = plt.figure(figsize=(13,13))
  
  row_count = -1
  fig, axs = plt.subplots(5, 5, figsize=(10, 10))
  fig.tight_layout()

  for idx, result in enumerate(data):

    # If 25 samples have been stored, break out of loop
    if idx > 24:
      break
        
    rgb_image = np.transpose(result['image'], (1, 2, 0)) / 2 + 0.5
    label = result['label'].item()
    prediction = result['prediction'].item()

    # Plot image
    if idx % 5 == 0:
      row_count += 1
    axs[row_count][idx % 5].axis('off')
    axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
    axs[row_count][idx % 5].imshow(rgb_image)
    
  # save the plot
  plt.savefig(filename)
  files.download(filename)
