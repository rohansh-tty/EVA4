# S11 Assignment

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

 

**Assignment Solution**: ![S12 Assignment Solution]()



## **Model Features:**

1. Used GPU as Device.
2. CNN Type: ![ResNet18]()
3. Total Params: 6,574,090
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding Image Augmentation Techniques like RandomCrop, Flip & Cutout.  
6. Max Learning Rate: using ReduceLROnPlateau
7. Used NLLLoss() to calculate loss value.
8. Ran the model for 24 Epochs 

        * Highest Validation Accuracy: %
        
9. GradCam for 25 Misclassified Images.


## **Library Documentation:**

1.![AlbTestTransforms.py]() : Applies required image transformation to Test dataset using Albumentations library.

2.![AlbTrainTransforms.py]() : Applies required image transformation to Train dataset using Albumentations library.

3.![customNet.py](): Consists of main S11 CNN Model.

4.![execute.py](): Scripts to Test & Train the model.

5.![DataLoaders.py](): Scripts to load the datasets.

6.![displayData.py](): Consists of helper functions to plot images from dataset & misclassified images.

7.![rohan_library](): Imports all the required libraries at once.

8.![Gradcam](): Consists of Gradcam class & other related functions.

9.![LR Finder](): LR finder using FastAI Approach.

10.![cyclicLR](): Consists helper functions related to CycliclR.


## **Cyclic Plot**
![]()

## **Misclassified Images**
![]()

## **GradCam for Misclassified Images**
![]()


## Model Performance on Train & Test Data
![]()

![]()

## OneCycleLR Plot
![]()


## Model Logs


