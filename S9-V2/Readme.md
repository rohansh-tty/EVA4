# S9 Assignment

Task: 

    Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
    Please make sure that your test_transforms are simple and only using ToTensor and Normalize
    Implement GradCam function as a module. 
    Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
    Target Accuracy is 87%
    Submit answers to S9-Assignment-Solution. 

**Assignment Solution**: ![S9 Assignment Solution](https://github.com/Gilf641/EVA4/blob/master/S9-V2/S9Final_V2.ipynb)

## **Model Features:**

1. Used GPU as Device
2. ResNet Variant: ResNet18
3. Total Params: 11,173,962
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding few Image Augmentation Techniques like Rotation, HorizonatalFlip & Vertical Flip.  
6. Used CrossEntropyLoss() to calculate loss value.
7. Ran the model for 15 Epochs with 

        * Highest Train Accuracy: 92.56% 

        * Corresponding Test Accuracy: 88.13% 
8. Implemented GradCam using this as a ![reference](https://github.com/GunhoChoi/Grad-CAM-Pytorch)

* **Model Analysis:**
1. Lot of fluctuations in Validation Loss values. 
2. Clearly an Overfit Model.

* **Future Work**
1. Fix Overfitting by adding some more training data.


## **Library Documentation:**

1.![AlbTestTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/AlbTestTransforms.py) : Applies required image transformation to Test dataset using Albumentations library.

2.![AlbTrainTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/AlbTrainTransforms.py) : Applies required image transformation to Train dataset using Albumentations library.

3.![resNet.py](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/resNet.py): Consists of ResNet variants

4.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/execute.py): Scripts to Test & Train the model.

5.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/DataLoaders.py): Scripts to load the dataloaders.

6.![displayData.py](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/visualizeData.py): Consists of helper functions to plot images from dataset & misclassified images

7.![rohan_library](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/rohan_library.py): Imports all the required libraries at once.

8.![Gradcam](https://github.com/Gilf641/EVA4/blob/master/S9-V2/evaLibrary/Gradcam.py): Consists of Gradcam class & other related functions.



## **Misclassified Images**

![](https://github.com/Gilf641/EVA4/blob/master/S9-V2/Misclassfied.png)


