# S11 Assignment

Task: 


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
       

**Assignment Solution**: ![S11 Assignment Solution](https://github.com/Gilf641/EVA4/blob/master/S11/S11_Assignment.ipynb)



## **Model Features:**

1. Used GPU as Device.
2. CNN Type: ![customNet](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/customNet.py)
3. Total Params: 6,574,090
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding Image Augmentation Techniques like RandomCrop, Flip & Cutout.  
6. Max Learning Rate: Between 0.03 & 0.04
7. Used NLLLoss() to calculate loss value.
8. Ran the model for 24 Epochs 

        * Highest Validation Accuracy: 91.58%
        
9. Used One Cycle Policy where I started with lr = min lr at the First epoch and linearly increased it till max lr(sixth epoch). From there again decreased lr linearly.
10. Plotted Cyclic Learning Rate for the Training phase, with Learning Rate linearly increasing from Epoch 1 to Epoch 5 i.e min lr to max lr and then later it gets lowered to a min lr.
11. GradCam for 25 Misclassified Images.


## **Library Documentation:**

1.![AlbTestTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/AlbTestTransforms.py) : Applies required image transformation to Test dataset using Albumentations library.

2.![AlbTrainTransforms.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/AlbTrainTransforms.py) : Applies required image transformation to Train dataset using Albumentations library.

3.![customNet.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/customNet.py): Consists of main S11 CNN Model.

4.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/execute.py): Scripts to Test & Train the model.

5.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/DataLoaders.py): Scripts to load the datasets.

6.![displayData.py](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/displayData.py): Consists of helper functions to plot images from dataset & misclassified images.

7.![rohan_library](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/rohan_library.py): Imports all the required libraries at once.

8.![Gradcam](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/Gradcam.py): Consists of Gradcam class & other related functions.

9.![LR Finder](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/LR_Finder.py): LR finder using FastAI Approach.

10.![cyclicLR](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/cyclicLR.py): Consists helper functions related to CycliclR.


## **Cyclic Plot**
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/clr_plot.png)

## **Misclassified Images**
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/MiscImages.png)

## **GradCam for Misclassified Images**
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/GradCamPlot.png)


## Model Performance on Train & Test Data
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/AccPlot-V3.png)

![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/LossPlot-V3.png)

## OneCycleLR Plot
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/OCP.png)


## Model Logs



  0%|          | 0/98 [00:00<?, ?it/s]

EPOCH:  1

/content/drive/My Drive/EVA4/updLib2/evaLibrary/customNet.py:122: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(outX)
Loss=2.7940001487731934 Batch_id=97 Accuracy=32.20: 100%|██████████| 98/98 [00:25<00:00,  3.85it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (inf --> 2.1795).  Saving model ...

Test set: Average loss: 2.1795, Accuracy: 3780/10000 (37.80%)

Learning Rate = 0.00665 for EPOCH 2
EPOCH:  2

Loss=1.7731581926345825 Batch_id=97 Accuracy=48.31: 100%|██████████| 98/98 [00:25<00:00,  3.81it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (2.1795 --> 1.1940).  Saving model ...

Test set: Average loss: 1.1940, Accuracy: 5512/10000 (55.12%)

Learning Rate = 0.01241 for EPOCH 3
EPOCH:  3

Loss=1.4110983610153198 Batch_id=97 Accuracy=56.27: 100%|██████████| 98/98 [00:26<00:00,  3.68it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (1.1940 --> 1.1878).  Saving model ...

Test set: Average loss: 1.1878, Accuracy: 5929/10000 (59.29%)

Learning Rate = 0.01818 for EPOCH 4
EPOCH:  4

Loss=1.3319900035858154 Batch_id=97 Accuracy=62.02: 100%|██████████| 98/98 [00:26<00:00,  3.69it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (1.1878 --> 1.0625).  Saving model ...

Test set: Average loss: 1.0625, Accuracy: 6410/10000 (64.10%)

Learning Rate = 0.02394 for EPOCH 5
EPOCH:  5

Loss=1.0636788606643677 Batch_id=97 Accuracy=66.46: 100%|██████████| 98/98 [00:27<00:00,  3.63it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (1.0625 --> 0.8815).  Saving model ...

Test set: Average loss: 0.8815, Accuracy: 6992/10000 (69.92%)

Learning Rate = 0.02971 for EPOCH 6
EPOCH:  6

Loss=0.9335458874702454 Batch_id=97 Accuracy=70.97: 100%|██████████| 98/98 [00:27<00:00,  3.54it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.8815 --> 0.7325).  Saving model ...

Test set: Average loss: 0.7325, Accuracy: 7482/10000 (74.82%)

Learning Rate = 0.02856 for EPOCH 7
EPOCH:  7

Loss=0.8790552616119385 Batch_id=97 Accuracy=74.29: 100%|██████████| 98/98 [00:27<00:00,  3.63it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.7325 --> 0.7048).  Saving model ...

Test set: Average loss: 0.7048, Accuracy: 7508/10000 (75.08%)

Learning Rate = 0.02705 for EPOCH 8
EPOCH:  8

Loss=0.7754079103469849 Batch_id=97 Accuracy=77.35: 100%|██████████| 98/98 [00:27<00:00,  3.56it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.7790, Accuracy: 7700/10000 (77.00%)

Learning Rate = 0.02553 for EPOCH 9
EPOCH:  9

Loss=0.7494895458221436 Batch_id=97 Accuracy=79.12: 100%|██████████| 98/98 [00:27<00:00,  3.52it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.7975, Accuracy: 7618/10000 (76.18%)

Learning Rate = 0.02402 for EPOCH 10
EPOCH:  10

Loss=0.6484556198120117 Batch_id=97 Accuracy=80.83: 100%|██████████| 98/98 [00:27<00:00,  3.55it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.7048 --> 0.6828).  Saving model ...

Test set: Average loss: 0.6828, Accuracy: 7826/10000 (78.26%)

Learning Rate = 0.0225 for EPOCH 11
EPOCH:  11

Loss=0.6594603061676025 Batch_id=97 Accuracy=82.25: 100%|██████████| 98/98 [00:28<00:00,  3.44it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.7051, Accuracy: 7647/10000 (76.47%)

Learning Rate = 0.02099 for EPOCH 12
EPOCH:  12

Loss=0.5890487432479858 Batch_id=97 Accuracy=82.56: 100%|██████████| 98/98 [00:28<00:00,  3.48it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.6828 --> 0.5586).  Saving model ...

Test set: Average loss: 0.5586, Accuracy: 7756/10000 (77.56%)

Learning Rate = 0.01948 for EPOCH 13
EPOCH:  13

Loss=0.6300306916236877 Batch_id=97 Accuracy=83.71: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.6039, Accuracy: 7951/10000 (79.51%)

Learning Rate = 0.01796 for EPOCH 14
EPOCH:  14

Loss=0.6027886867523193 Batch_id=97 Accuracy=84.13: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.5586 --> 0.5360).  Saving model ...

Test set: Average loss: 0.5360, Accuracy: 8135/10000 (81.35%)

Learning Rate = 0.01645 for EPOCH 15
EPOCH:  15

Loss=0.5457993745803833 Batch_id=97 Accuracy=84.89: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.5360 --> 0.4963).  Saving model ...

Test set: Average loss: 0.4963, Accuracy: 8135/10000 (81.35%)

Learning Rate = 0.01493 for EPOCH 16
EPOCH:  16

Loss=0.5552917122840881 Batch_id=97 Accuracy=85.72: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.4963 --> 0.4533).  Saving model ...

Test set: Average loss: 0.4533, Accuracy: 8132/10000 (81.32%)

Learning Rate = 0.01342 for EPOCH 17
EPOCH:  17

Loss=0.5558308959007263 Batch_id=97 Accuracy=86.03: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.4839, Accuracy: 8354/10000 (83.54%)

Learning Rate = 0.0119 for EPOCH 18
EPOCH:  18

Loss=0.506567656993866 Batch_id=97 Accuracy=86.72: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.4533 --> 0.4161).  Saving model ...

Test set: Average loss: 0.4161, Accuracy: 8448/10000 (84.48%)

Learning Rate = 0.01039 for EPOCH 19
EPOCH:  19

Loss=0.5127120018005371 Batch_id=97 Accuracy=86.98: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.4161 --> 0.4048).  Saving model ...

Test set: Average loss: 0.4048, Accuracy: 8597/10000 (85.97%)

Learning Rate = 0.00888 for EPOCH 20
EPOCH:  20

Loss=0.5062524676322937 Batch_id=97 Accuracy=87.78: 100%|██████████| 98/98 [00:28<00:00,  3.46it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.4048 --> 0.3540).  Saving model ...

Test set: Average loss: 0.3540, Accuracy: 8556/10000 (85.56%)

Learning Rate = 0.00736 for EPOCH 21
EPOCH:  21

Loss=0.4368305206298828 Batch_id=97 Accuracy=88.58: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.3540 --> 0.3378).  Saving model ...

Test set: Average loss: 0.3378, Accuracy: 8742/10000 (87.42%)

Learning Rate = 0.00585 for EPOCH 22
EPOCH:  22

Loss=0.46277084946632385 Batch_id=97 Accuracy=89.50: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Test set: Average loss: 0.3541, Accuracy: 8703/10000 (87.03%)

Learning Rate = 0.00433 for EPOCH 23
EPOCH:  23

Loss=0.39608022570610046 Batch_id=97 Accuracy=90.64: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.3378 --> 0.2969).  Saving model ...

Test set: Average loss: 0.2969, Accuracy: 9013/10000 (90.13%)

Learning Rate = 0.00282 for EPOCH 24
EPOCH:  24

Loss=0.29732298851013184 Batch_id=97 Accuracy=92.26: 100%|██████████| 98/98 [00:28<00:00,  3.47it/s]

Validation loss has  decreased (0.2969 --> 0.2166).  Saving model ...

Test set: Average loss: 0.2166, Accuracy: 9158/10000 (91.58%)

Learning Rate = 0.0013 for EPOCH 25


