# S10 Assignment

Task: 
Assignment: 

    Pick your last code
    Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
    Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.) 
        Move LR Finder code to your modules
        Implement LR Finder (for SGD, not for ADAM)
        Implement ![ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
    Find best LR to train your model
    Use SDG with Momentum
    Train for 50 Epochs. 
    Show Training and Test Accuracy curves
    Target 88% Accuracy.
    Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
    Submit


**Assignment Solution**: ![S10 Assignment Solution]()

## **Model Features:**

1. Used GPU as Device
2. ResNet Variant: ResNet18
3. Total Params: 11,173,962
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding Image Augmentation Techniques like Rotation, HorizonatalFlip, Vertical Flip & Cutout.  
6. Used CrossEntropyLoss() to calculate loss value.
7. Ran the model for 50 Epochs with 

        * Highest Validation Accuracy: 88.91% (50th Epoch)
        
8. Implemented GradCam.
9. Plotted GradCam for 25 Misclassified Images

* **Model Analysis:**
1. Lot of fluctuations in Validation Loss values. 
2. Clearly an Overfit Model.

* **Future Work**
1. Fix Overfitting by adding some more training data.


## **Library Documentation:**

1.![AlbTestTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/AlbTestTransforms.py) : Applies required image transformation to Test dataset using Albumentations library.

2.![AlbTrainTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/AlbTrainTransforms.py) : Applies required image transformation to Train dataset using Albumentations library.

3.![resNet.py](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/resNet.py): Consists of ResNet variants

4.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/execute.py): Scripts to Test & Train the model.

5.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/DataLoaders.py): Scripts to load the dataloaders.

6.![displayData.py](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/visualizeData.py): Consists of helper functions to plot images from dataset & misclassified images

7.![rohan_library](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/rohan_library.py): Imports all the required libraries at once.

8.![Gradcam](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/Gradcam.py): Consists of Gradcam class & other related functions.

9.![LR Finder](https://github.com/Gilf641/EVA4/blob/master/S10/evaLibrary/LRFinder.py): LR finder using FastAI Approach

## **Misclassified Images**

![](https://github.com/Gilf641/EVA4/blob/master/S10/Misclassfied.png)


