# S9 Assignment( 3rd Version)

Task: 

    Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
    Please make sure that your test_transforms are simple and only using ToTensor and Normalize
    Implement GradCam function as a module. 
    Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
    Target Accuracy is 87%
    Submit answers to S9-Assignment-Solution. 

**Assignment Solution**: ![S9 Assignment Solution](https://github.com/Gilf641/EVA4/blob/master/S9-V3/S9_AssignmentSolution.ipynb)

## **Model Features:**

1. Used GPU as Device
2. ResNet Variant: ResNet18
3. Total Params: 11,173,962
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding few Image Augmentation Techniques like **PadIfNeeded, RandomCrop, Rotate and Cutout**.  
6. Used CrossEntropyLoss() to calculate loss value.
7. Used **GhostBatchNormalization** instead of Normal BatchNormalization.
8. Compared two versions of ResNet18, one with Nesterov Momentum and one without.
9. Ran both of them model for 30 Epochs with 
       
      **ResNet18 with Nesterov**

        * Highest Train Accuracy: 93%

        * Corresponding Test Accuracy: 88.50% 
        
      **ResNet18 without Nesterov**
      
        * Highest Train Accuracy: 93.73%

        * Corresponding Test Accuracy: 89.22% 
        
        
10. Implemented GradCam using this as a ![reference](https://github.com/GunhoChoi/Grad-CAM-Pytorch)

* **Model Analysis:**
1. Lot of fluctuations in Validation Loss values. 
2. Both of the models are slightly Overfit.
 

## **Library Documentation:**

1.![AlbTestTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/AlbTestTransforms.py) : Applies required image transformation to Test dataset using Albumentations library.

2.![AlbTrainTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/AlbTrainTransforms.py) : Applies required image transformation to Train dataset using Albumentations library.

3.![resNet.py](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/resNet.py): Consists of ResNet variants

4.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/execute.py): Scripts to Test & Train the model.

5.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/DataLoaders.py): Scripts to load the dataloaders.

6.![displayData.py](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/visualizeData.py): Consists of helper functions to plot images from dataset, misclassified images, accuracy and loss curves

7.![rohan_library](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/rohan_library.py): Imports all the required libraries at once.

8.![Gradcam](https://github.com/Gilf641/EVA4/blob/master/S9-V3/evaLibrary/Gradcam.py): Consists of Gradcam class & other related functions.


## **Misclassified Images**


![]()


## Model Comparision


![with Nesterov](https://github.com/Gilf641/EVA4/blob/master/S9-V3/Images/AccPlot(withNest30).png)
![without Nesterov](https://github.com/Gilf641/EVA4/blob/master/S9-V3/Images/AccPlot(withoutNest30).png)


With accuracy as metric, I found that the model without Nesterov slightly performed better than one with Nesterov. Kinda confused here. 



## GradCam Plot


![](https://github.com/Gilf641/EVA4/blob/master/S9-V3/Images/GradCam-Plot.png)



## Model Logs(without Nesterov Momentum)
  0%|          | 0/391 [00:00<?, ?it/s]

<class 'int'>
EPOCH:  1

Loss=2.3993778228759766 Batch_id=390 Accuracy=43.73: 100%|██████████| 391/391 [00:56<00:00,  6.89it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (inf --> 1.1494).  Saving model ...

Test set: Average loss: 1.1494, Accuracy: 5593/10000 (55.93%)

EPOCH:  2

Loss=1.8267278671264648 Batch_id=390 Accuracy=60.35: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (1.1494 --> 1.0869).  Saving model ...

Test set: Average loss: 1.0869, Accuracy: 6460/10000 (64.60%)

EPOCH:  3

Loss=1.815322995185852 Batch_id=390 Accuracy=69.18: 100%|██████████| 391/391 [00:58<00:00,  6.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 1.2238, Accuracy: 7265/10000 (72.65%)

EPOCH:  4

Loss=1.6664260625839233 Batch_id=390 Accuracy=74.10: 100%|██████████| 391/391 [00:58<00:00,  6.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (1.0869 --> 0.5879).  Saving model ...

Test set: Average loss: 0.5879, Accuracy: 7737/10000 (77.37%)

EPOCH:  5

Loss=1.612652063369751 Batch_id=390 Accuracy=77.21: 100%|██████████| 391/391 [00:58<00:00,  6.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (0.5879 --> 0.5309).  Saving model ...

Test set: Average loss: 0.5309, Accuracy: 7980/10000 (79.80%)

EPOCH:  6

Loss=1.5937676429748535 Batch_id=390 Accuracy=79.69: 100%|██████████| 391/391 [00:58<00:00,  6.70it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (0.5309 --> 0.1877).  Saving model ...

Test set: Average loss: 0.1877, Accuracy: 8223/10000 (82.23%)

EPOCH:  7

Loss=1.332169771194458 Batch_id=390 Accuracy=81.63: 100%|██████████| 391/391 [00:58<00:00,  6.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4531, Accuracy: 8301/10000 (83.01%)

EPOCH:  8

Loss=1.3006327152252197 Batch_id=390 Accuracy=83.16: 100%|██████████| 391/391 [00:58<00:00,  6.66it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.2527, Accuracy: 8298/10000 (82.98%)

EPOCH:  9

Loss=1.274045467376709 Batch_id=390 Accuracy=84.50: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5480, Accuracy: 8530/10000 (85.30%)

EPOCH:  10

Loss=1.4447596073150635 Batch_id=390 Accuracy=85.31: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4435, Accuracy: 8338/10000 (83.38%)

EPOCH:  11

Loss=1.6077896356582642 Batch_id=390 Accuracy=86.46: 100%|██████████| 391/391 [00:58<00:00,  6.70it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.7754, Accuracy: 8404/10000 (84.04%)

EPOCH:  12

Loss=1.3639236688613892 Batch_id=390 Accuracy=86.71: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.2870, Accuracy: 8484/10000 (84.84%)

EPOCH:  13

Loss=1.3148407936096191 Batch_id=390 Accuracy=88.03: 100%|██████████| 391/391 [00:58<00:00,  6.70it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5322, Accuracy: 8624/10000 (86.24%)

EPOCH:  14

Loss=0.9593429565429688 Batch_id=390 Accuracy=88.29: 100%|██████████| 391/391 [00:58<00:00,  6.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5626, Accuracy: 8654/10000 (86.54%)

EPOCH:  15

Loss=1.0032812356948853 Batch_id=390 Accuracy=89.32: 100%|██████████| 391/391 [00:58<00:00,  6.70it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.3430, Accuracy: 8595/10000 (85.95%)

EPOCH:  16

Loss=1.1367262601852417 Batch_id=390 Accuracy=90.00: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.6502, Accuracy: 8548/10000 (85.48%)

EPOCH:  17

Loss=1.1067752838134766 Batch_id=390 Accuracy=90.30: 100%|██████████| 391/391 [00:58<00:00,  6.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4447, Accuracy: 8755/10000 (87.55%)

EPOCH:  18

Loss=0.9995589256286621 Batch_id=390 Accuracy=90.38: 100%|██████████| 391/391 [00:58<00:00,  6.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4332, Accuracy: 8622/10000 (86.22%)

EPOCH:  19

Loss=1.0414959192276 Batch_id=390 Accuracy=91.10: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.3009, Accuracy: 8627/10000 (86.27%)

EPOCH:  20

Loss=0.9841194152832031 Batch_id=390 Accuracy=91.44: 100%|██████████| 391/391 [00:58<00:00,  6.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.6172, Accuracy: 8747/10000 (87.47%)

EPOCH:  21

Loss=1.029988408088684 Batch_id=390 Accuracy=91.50: 100%|██████████| 391/391 [00:58<00:00,  6.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.5387, Accuracy: 8727/10000 (87.27%)

EPOCH:  22

Loss=0.9492353200912476 Batch_id=390 Accuracy=91.94: 100%|██████████| 391/391 [00:58<00:00,  6.70it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4468, Accuracy: 8759/10000 (87.59%)

EPOCH:  23

Loss=0.9553390741348267 Batch_id=390 Accuracy=92.18: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]

Validation loss has  decreased (0.1877 --> 0.0348).  Saving model ...

Test set: Average loss: 0.0348, Accuracy: 8548/10000 (85.48%)

EPOCH:  24

Loss=1.022128701210022 Batch_id=390 Accuracy=92.53: 100%|██████████| 391/391 [00:58<00:00,  6.70it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.6076, Accuracy: 8681/10000 (86.81%)

EPOCH:  25

Loss=0.8300677537918091 Batch_id=390 Accuracy=92.78: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.1909, Accuracy: 8780/10000 (87.80%)

EPOCH:  26

Loss=0.9589564800262451 Batch_id=390 Accuracy=92.75: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.4901, Accuracy: 8781/10000 (87.81%)

EPOCH:  27

Loss=0.8555641174316406 Batch_id=390 Accuracy=93.21: 100%|██████████| 391/391 [00:58<00:00,  6.69it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.7185, Accuracy: 8786/10000 (87.86%)

EPOCH:  28

Loss=0.9609363079071045 Batch_id=390 Accuracy=93.32: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.1905, Accuracy: 8831/10000 (88.31%)

EPOCH:  29

Loss=0.7528522610664368 Batch_id=390 Accuracy=93.30: 100%|██████████| 391/391 [00:58<00:00,  6.68it/s]
  0%|          | 0/391 [00:00<?, ?it/s]


Test set: Average loss: 0.1787, Accuracy: 8703/10000 (87.03%)

EPOCH:  30

Loss=0.7604323625564575 Batch_id=390 Accuracy=93.73: 100%|██████████| 391/391 [00:58<00:00,  6.67it/s]


Test set: Average loss: 0.4899, Accuracy: 8922/10000 (89.22%)



