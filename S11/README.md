# S11 Assignment

Task: 
Assignment: 

    Assignment:

    1. Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that
        11s11.png
    2. Write a code which
        uses this new ResNet Architecture for Cifar10:
            PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
            Layer1 -
                X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
                R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
                Add(X, R1)
            Layer 2 -
                Conv 3x3 [256k]
                MaxPooling2D
                BN
                ReLU
            Layer 3 -
                X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
                R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
                Add(X, R2)
            MaxPooling with Kernel Size 4
            FC Layer 
            SoftMax
       3. Uses One Cycle Policy such that:
            Total Epochs = 24
            Max at Epoch = 5
            LRMIN = FIND
            LRMAX = FIND
            NO Annihilation
        4. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
        5. Batch size = 512
        6. Target Accuracy: 90%. 
       

**Assignment Solution**: ![S11 Assignment Solution]()



## **Model Features:**

1. Used GPU as Device
2. CNN Type: ![customNet]()
3. Total Params: 6,574,090
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding Image Augmentation Techniques like RandomCrop, Flip & Cutout.  
6. Max Learning Rate: Between 0.2 & 0.3
7. Used NLLLoss() to calculate loss value.
8. Ran the model for 24 Epochs 

        * Highest Validation Accuracy: 
        
9. Used One Cycle Policy where I started with lr = min lr at the First epoch and linearly increased it till max lr(sixth epoch). From there again decreased lr linearly.
10. 
11. Plotted GradCam for 25 Misclassified Images

## **Library Documentation:**

1.![AlbTestTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/AlbTestTransforms.py) : Applies required image transformation to Test dataset using Albumentations library.

2.![AlbTrainTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/AlbTrainTransforms.py) : Applies required image transformation to Train dataset using Albumentations library.

3.![customNet.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/customNet.py): Consists of main CNN Model.

4.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/execute.py): Scripts to Test & Train the model.

5.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/DataLoaders.py): Scripts to load the dataloaders.

6.![displayData.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/displayData.py): Consists of helper functions to plot images from dataset & misclassified images.

7.![rohan_library](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/rohan_library.py): Imports all the required libraries at once.

8.![Gradcam](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/Gradcam.py): Consists of Gradcam class & other related functions.

9.![LR Finder](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/LR_Finder.py): LR finder using FastAI Approach.

10.![cyclicLR](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/cyclicLR.py): Consists helper functions related to CycliclR.

## **Misclassified Images**
![]()
## **GradCam for Misclassified Images**
![](https://github.com/Gilf641/EVA4/blob/master/S10/Images/GradCamPlot.jpeg)


## Model Performance on Train & Test Data
![]()
![]()

## Model Logs
