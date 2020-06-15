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


## Model Logs


  0%|          | 0/391 [00:00<?, ?it/s]

EPOCH:  1

Loss=2.3382341861724854 Batch_id=390 Accuracy=42.57: 100%|██████████| 391/391 [00:29<00:00, 13.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (inf --> 1.5662).  Saving model ...

Test set: Average loss: 1.5662, Accuracy: 5048/10000 (50.48%)

EPOCH:  2

Loss=1.907501220703125 Batch_id=390 Accuracy=57.37: 100%|██████████| 391/391 [00:29<00:00, 13.17it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (1.5662 --> 1.0662).  Saving model ...

Test set: Average loss: 1.0662, Accuracy: 5537/10000 (55.37%)

EPOCH:  3

Loss=1.6751058101654053 Batch_id=390 Accuracy=65.38: 100%|██████████| 391/391 [00:29<00:00, 13.11it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (1.0662 --> 0.8210).  Saving model ...

Test set: Average loss: 0.8210, Accuracy: 6567/10000 (65.67%)

EPOCH:  4

Loss=1.646028995513916 Batch_id=390 Accuracy=70.21: 100%|██████████| 391/391 [00:29<00:00, 13.21it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 1.4249, Accuracy: 6878/10000 (68.78%)

EPOCH:  5

Loss=1.4175937175750732 Batch_id=390 Accuracy=74.03: 100%|██████████| 391/391 [00:29<00:00, 13.21it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.9876, Accuracy: 7436/10000 (74.36%)

EPOCH:  6

Loss=1.4975175857543945 Batch_id=390 Accuracy=75.86: 100%|██████████| 391/391 [00:29<00:00, 13.20it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 1.1863, Accuracy: 7336/10000 (73.36%)

EPOCH:  7

Loss=1.3047308921813965 Batch_id=390 Accuracy=78.03: 100%|██████████| 391/391 [00:29<00:00, 13.23it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.9541, Accuracy: 7468/10000 (74.68%)

EPOCH:  8

Loss=1.0850780010223389 Batch_id=390 Accuracy=79.40: 100%|██████████| 391/391 [00:29<00:00, 13.19it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (0.8210 --> 0.2847).  Saving model ...

Test set: Average loss: 0.2847, Accuracy: 7729/10000 (77.29%)

EPOCH:  9

Loss=1.1919806003570557 Batch_id=390 Accuracy=81.08: 100%|██████████| 391/391 [00:29<00:00, 13.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 1.0568, Accuracy: 7681/10000 (76.81%)

EPOCH:  10

Loss=1.0741041898727417 Batch_id=390 Accuracy=82.47: 100%|██████████| 391/391 [00:29<00:00, 13.12it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (0.2847 --> 0.2683).  Saving model ...

Test set: Average loss: 0.2683, Accuracy: 7926/10000 (79.26%)

EPOCH:  11

Loss=0.9234122037887573 Batch_id=390 Accuracy=84.15: 100%|██████████| 391/391 [00:29<00:00, 13.19it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.3667, Accuracy: 8160/10000 (81.60%)

EPOCH:  12

Loss=0.7548419237136841 Batch_id=390 Accuracy=86.07: 100%|██████████| 391/391 [00:29<00:00, 13.14it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 1.1275, Accuracy: 8483/10000 (84.83%)

EPOCH:  13

Loss=0.8244100213050842 Batch_id=390 Accuracy=88.47: 100%|██████████| 391/391 [00:29<00:00, 13.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.2941, Accuracy: 8564/10000 (85.64%)

EPOCH:  14

Loss=0.6259176731109619 Batch_id=390 Accuracy=91.01: 100%|██████████| 391/391 [00:29<00:00, 13.20it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Test set: Average loss: 0.5218, Accuracy: 8798/10000 (87.98%)

EPOCH:  15

Loss=0.754241406917572 Batch_id=390 Accuracy=92.56: 100%|██████████| 391/391 [00:29<00:00, 13.14it/s]

Test set: Average loss: 0.3434, Accuracy: 8813/10000 (88.13%)



