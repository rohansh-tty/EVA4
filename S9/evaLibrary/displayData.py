
#Visualization Functions
import matplotlib.pyplot as plt
import numpy as np
import torch

channel_means = (0.5, 0.5, 0.5)
channel_stdevs = (0.5, 0.5, 0.5)


def unnormalize(img):
  img = img.numpy().astype(dtype=np.float32)
  
  for i in range(img.shape[0]):
    img[i] = (img[i]*channel_stdevs[i])+channel_means[i] # if not unnormalized then the resulting images will be dark and not visible
  return np.transpose(img, (1,2,0))

class_names = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

def classImages(dataiterator):
  num_classes = 10
  # display 10 images from each category. 
  images, labels = iter(dataiterator).next()
  # channel_means = (0.5, 0.5, 0.5)
  # channel_stdevs = (0.5, 0.5, 0.5)

  r, c = 10, 10
  n = 5
  fig = plt.figure(figsize=(14,14))
  fig.subplots_adjust(hspace=0.01, wspace=0.01)
  for i in range(num_classes):
    idx = np.random.choice(np.where(labels[:]==i)[0], n)
    ax = plt.subplot(r, c, i*c+1) # (10, 10, i*10+1)
    ax.text(-1.5, 0.5, class_names[i], fontsize=14)
    plt.axis('off')
    for j in range(1, n+1):
      plt.subplot(r, c, i*c+j+1) # (10, 10, i*10+j+1)
      plt.imshow(unnormalize(images[idx[j-1]]), interpolation='none')
      plt.axis('off')
  plt.show()


# def imshow(img):
#   img = img / 2 + 0.5  # unnormalize
#   plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# def randomImages(dataiterator, classes):
#         images, labels = dataiterator.next()
#         images = images.numpy()  # convert images to numpy for display

#         # plot the images in the batch, along with the corresponding labels
#         fig = plt.figure(figsize=(25, 4))
#         # display 20 images
#         for i in range(len(classes)):
#           # for idx in np.arange(5):          
#             ax = fig.add_subplot(2, 20 / 2, i + 1, xticks=[], yticks=[])
#             imshow(images[i])
#             ax.set_title(classes[labels[i]])

# randomImages(dataiter,classes)





# Display Images from training dataset
def plotImage(img):
    img = img / 2 + 0.5  # unnormalize this is make sure the image is visible, if this step is skipped then the resulting images have a dark portion
    npimg = img.numpy()   # converting image to numpy array format
    plt.imshow(np.transpose(npimg, (1, 2, 0)))    # transposing npimg array


# Misclassified Images
from google.colab import files
def misclassifiedOnes(model, testLoader, data,filename):

  #model: ModelName
  #data: Incorrect Classes in Test() of Test_Train class
  #filename: Pass on the filename with which you want to save misclassified images
  
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # classs names in the dataset

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = model.to(device)
  dataiter = iter(testLoader) 
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


# Correctly Classified Images
from google.colab import files
def correctlyClassifed(model, testLoader, data,filename):

  #model: ModelName
  #data: Correct Classes in Test() of Test_Train class
  #filename: Pass on the filename with which you want to save misclassified images
  
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # classs names in the dataset

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = model.to(device)
  dataiter = iter(testLoader) 
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
