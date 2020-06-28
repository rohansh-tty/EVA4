# S12 Assignment

Task: 


    Assignment:
    
    Assignment A:
        Download this TINY IMAGENET (Links to an external site.) dataset. 
        Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
        Submit Results. Of course, you are using your own package for everything. You can look at this (Links to an external site.) for reference. 
    Assignment B:
        Download 50 images of dogs. 
        Use this (Links to an external site.) to annotate bounding boxes around the dogs.
        Download JSON file. 
        Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work). 
        Refer to this tutorial (Links to an external site.). Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub. 

 

**Assignment A Solution**: ![S12 Assignment A Solution](https://github.com/Gilf641/EVA4/blob/master/S12/S12_AssignmentSolution.ipynb)

**Assignment B Solution**: ![S12 Assignment B Solution](https://github.com/Gilf641/EVA4/blob/master/S12/S12_AssignmentB(K-Means%20Clustering).ipynb)
* Details below ![]()


# Assignment A


## **Model Features:**

1. Used GPU as Device.
2. CNN Type: ResNet18
3. Total Params: 11,271,432
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding Image Augmentation Techniques like RandomCrop, Flip & Cutout.  
6. Max Learning Rate: 0.01
7. Used NLLLoss() to calculate loss value.
8. Ran the model for 50 Epochs 

        * Highest Validation Accuracy: 51.20%
        
9. GradCam for 25 Misclassified Images.


## **Library Documentation:**

1.![AlbTransforms.py]() : Applies required image transformation to both Train & Test dataset using Albumentations library.

2.![DataPrep.py](): Consists of Custom DataSet Class and some helper functions to apply transformations, extract classID etc.

3.![resNet.py](): Consists of main ResNet model

4.![execute.py](): Scripts to Test & Train the model.

5.![DataLoaders.py](): Scripts to load the datasets.

6.![displayData.py](): Consists of helper functions to plot images from dataset & misclassified images.

7.![Gradcam.py](): Consists of Gradcam class & other related functions.

8.![LR Finder.py](): LR finder using FastAI Approach.

9.![cyclicLR.py](): Consists helper functions related to CycliclR.


## **LR Finder Plot**
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/LR%20finder.png)

## **Misclassified Images**
![](https://github.com/Gilf641/EVA4/tree/master/S12/Assignment-A/Images/Misclassified.png)

## **GradCam for Misclassified Images**
![](https://github.com/Gilf641/EVA4/tree/master/S12/Assignment-A/Images/GradCam.png)


## Model Performance on Train & Test Data
![](https://github.com/Gilf641/EVA4/tree/master/S12/Assignment-A/Images/AccPlot.png)

![](https://github.com/Gilf641/EVA4/tree/master/S12/Assignment-A/Images/LossPlot.png)



## ![Model Logs](https://github.com/Gilf641/EVA4/tree/master/S12/Assignment-A/ModelLogs.md)

# Assignment B 

![Images Folder]()
![JSON FILE]()
