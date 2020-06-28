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


## **Cyclic Plot**
![]()

## **Misclassified Images**
![](https://github.com/Gilf641/EVA4/tree/master/S12/Images/Misclassified.png)

## **GradCam for Misclassified Images**
![](https://github.com/Gilf641/EVA4/tree/master/S12/Images/GradCam.png)


## Model Performance on Train & Test Data
![](https://github.com/Gilf641/EVA4/tree/master/S12/Images/AccPlot.png)

![](https://github.com/Gilf641/EVA4/tree/master/S12/Images/LossPlot.png)

## OneCycleLR Plot
![]()


## ![Model Logs]()

# Assignment B 

![Images Folder]()
![JSON FILE]()

Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work)

File which helps in storing Data Structures & Objects in JavaScript Object Notation is called JSON File. It consists of key-value pairs similar to Python Dictionaries. 
Now in this JSON file there are around 4 Keys/Attributes i.e Filename, Size, Regions and Attributes

Example: 
> "dog11.jpg5671":{"filename":"dog11.jpg","size":5671,"regions":[{"shape_attributes":{"name":"rect","x":97,"y":29,"width":82,"height":74},"region_attributes":{"Class":"Dog"}}],"file_attributes":{"caption":"","public_domain":"no","image_url":""}}

1. First, the Image name which is same as original image name with size attached at the end. This is the main Key. 
2. For this we have around 1 Value, which a dictionary consisting of 4 Keys i.e Filename, Size, Regions and File Attributes.
3. FileName is again the Original Image Name. Simple
4. Size is the Image Size.
5. Region consists of two attributes Shape & Region Attribute. 
> Shape Attribute consists of 4 elements, which refer to the Bounding Box dimension. It consists of X, Y, W & H. 
    (X,Y) is the starting left (point/corner)coordinate of the bounding box, while W & H are width and height of the Bounding Box. Adding X to W & Y to H results in                 Bottom Right Coordinate of the Bounding Box.    

