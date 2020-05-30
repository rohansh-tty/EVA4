# S9 Assignment

Task: 

    Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
    Please make sure that your test_transforms are simple and only using ToTensor and Normalize
    Implement GradCam function as a module. 
    Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
    Target Accuracy is 87%
    Submit answers to S9-Assignment-Solution. 

**Assignment Solution**: 

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

## **Misclassified Images**
