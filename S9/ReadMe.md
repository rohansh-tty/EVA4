# S9 Assignment

Task: 

    Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
    Please make sure that your test_transforms are simple and only using ToTensor and Normalize
    Implement GradCam function as a module. 
    Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
    Target Accuracy is 87%
    Submit answers to S9-Assignment-Solution. 

**Assignment Solution**: ![S9 Assignment Solution](https://github.com/Gilf641/EVA4/blob/master/S9/S9Final.ipynb)

## **Model Features:**

1. Used GPU as Device
2. ResNet Variant: ResNet18
3. Total Params: 11,173,962
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding few Image Augmentation Techniques like Rotation, HorizonatalFlip & Vertical Flip.  
6. Used CrossEntropyLoss() to calculate loss value.
7. Ran the model for 15 Epochs with 

        * Highest Train Accuracy: 91.84% 

        * Corresponding Test Accuracy: 89.22% 
8. Implemented GradCam using this as a ![reference](https://github.com/GunhoChoi/Grad-CAM-Pytorch)

* **Model Analysis:**
1. Lot of fluctuations in Validation Loss values. 
2. Not that Overfit Model.
3. In Misclassified Images, one can see that most of images are either hidden / occluded / oriented in different way. Also in some images the class deciding portions is kinda dark. Eg: AirPlane Image (2nd Row, 4th Column) with it's wings, rear parts are not that visible. Front portion of Truck( 5th row, 2nd column)is excluded.






## **Library Documentation:**

1.![alb2.py](https://github.com/Gilf641/EVA4/blob/master/S9/evaLibrary/alb2.py) : Applies required image transformation to both Test & Train dataset using Albumentations library.

2.![resNet.py](https://github.com/Gilf641/EVA4/blob/master/S9/evaLibrary/resNet.py): Consists of ResNet variants

3.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S9/evaLibrary/execute.py): Scripts to Test & Train the model.

4.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S9/evaLibrary/DataLoaders.py): Scripts to load the dataloaders.

5.![displayData.py](https://github.com/Gilf641/EVA4/blob/master/S9/evaLibrary/visualizeData.py): Consists of helper functions to plot images from dataset & misclassified images

6.![rohan_library](https://github.com/Gilf641/EVA4/blob/master/S9/evaLibrary/rohan_library.py): Imports all the required libraries at once.

7.![Gradcam](https://github.com/Gilf641/EVA4/blob/master/S9/evaLibrary/Gradcam.py): Consists of Gradcam class & other related functions.



## **Misclassified Images**

![](https://github.com/Gilf641/EVA4/blob/master/S9/Misclassfied.png)
