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
       

**Assignment Solution**: ![S11 Assignment Solution](https://github.com/Gilf641/EVA4/blob/master/S11/S11_AssignmentSolution.ipynb)



## **Model Features:**

1. Used GPU as Device.
2. CNN Type: ![customNet](https://github.com/Gilf641/EVA4/blob/master/S11/evaLibrary/customNet.py)
3. Total Params: 6,574,090
4. Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.
5. Also Trained the model a bit harder by adding Image Augmentation Techniques like RandomCrop, Flip & Cutout.  
6. Max Learning Rate: Between 0.02 & 0.03
7. Used NLLLoss() to calculate loss value.
8. Ran the model for 24 Epochs 

        * Highest Validation Accuracy: 
        
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
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/Misclassified%20Images.png)


## **Misclassified Images**
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/clr_plot.png)
## **GradCam for Misclassified Images**
![](https://github.com/Gilf641/EVA4/blob/master/S10/Images/GradCamPlot.jpeg)


## Model Performance on Train & Test Data
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/accplots.png)
![](https://github.com/Gilf641/EVA4/blob/master/S11/Images/lossplots.png)

## Model Logs
  0%|          | 0/98 [00:00<?, ?it/s]

EPOCH:  1

/content/drive/My Drive/EVA4/updLib2/evaLibrary/customNet.py:122: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(outX)
Loss=1.5430012941360474 Batch_id=97 Accuracy=35.22: 100%|██████████| 98/98 [00:16<00:00,  5.99it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (inf --> 1.3590).  Saving model ...

Test set: Average loss: 1.3590, Accuracy: 4906/10000 (49.06%)

Learning Rate = 0.00674 for EPOCH 2
EPOCH:  2

Loss=1.2717164754867554 Batch_id=97 Accuracy=50.76: 100%|██████████| 98/98 [00:16<00:00,  5.98it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 1.5191, Accuracy: 5395/10000 (53.95%)

Learning Rate = 0.012 for EPOCH 3
EPOCH:  3

Loss=1.0271655321121216 Batch_id=97 Accuracy=59.41: 100%|██████████| 98/98 [00:16<00:00,  5.96it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (1.3590 --> 0.9671).  Saving model ...

Test set: Average loss: 0.9671, Accuracy: 6707/10000 (67.07%)

Learning Rate = 0.01726 for EPOCH 4
EPOCH:  4

Loss=0.9532467722892761 Batch_id=97 Accuracy=65.11: 100%|██████████| 98/98 [00:16<00:00,  5.98it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 1.0276, Accuracy: 6363/10000 (63.63%)

Learning Rate = 0.02252 for EPOCH 5
EPOCH:  5

Loss=1.1825652122497559 Batch_id=97 Accuracy=68.81: 100%|██████████| 98/98 [00:16<00:00,  6.04it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 2.1322, Accuracy: 6693/10000 (66.93%)

Learning Rate = 0.02779 for EPOCH 6
EPOCH:  6

Loss=0.7077917456626892 Batch_id=97 Accuracy=72.83: 100%|██████████| 98/98 [00:16<00:00,  5.94it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.9671 --> 0.6560).  Saving model ...

Test set: Average loss: 0.6560, Accuracy: 7649/10000 (76.49%)

Learning Rate = 0.0266 for EPOCH 7
EPOCH:  7

Loss=0.7320030331611633 Batch_id=97 Accuracy=76.69: 100%|██████████| 98/98 [00:16<00:00,  5.93it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.6560 --> 0.6244).  Saving model ...

Test set: Average loss: 0.6244, Accuracy: 7870/10000 (78.70%)

Learning Rate = 0.02514 for EPOCH 8
EPOCH:  8

Loss=0.5680533647537231 Batch_id=97 Accuracy=79.11: 100%|██████████| 98/98 [00:16<00:00,  5.99it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.6884, Accuracy: 8002/10000 (80.02%)

Learning Rate = 0.02368 for EPOCH 9
EPOCH:  9

Loss=0.5387528538703918 Batch_id=97 Accuracy=81.00: 100%|██████████| 98/98 [00:16<00:00,  5.93it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.6244 --> 0.5786).  Saving model ...

Test set: Average loss: 0.5786, Accuracy: 8213/10000 (82.13%)

Learning Rate = 0.02222 for EPOCH 10
EPOCH:  10

Loss=0.4227910339832306 Batch_id=97 Accuracy=82.49: 100%|██████████| 98/98 [00:16<00:00,  5.97it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.5786 --> 0.3895).  Saving model ...

Test set: Average loss: 0.3895, Accuracy: 8424/10000 (84.24%)

Learning Rate = 0.02076 for EPOCH 11
EPOCH:  11

Loss=0.5307521820068359 Batch_id=97 Accuracy=83.87: 100%|██████████| 98/98 [00:16<00:00,  5.99it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.4690, Accuracy: 8351/10000 (83.51%)

Learning Rate = 0.01931 for EPOCH 12
EPOCH:  12

Loss=0.3876503109931946 Batch_id=97 Accuracy=85.08: 100%|██████████| 98/98 [00:16<00:00,  5.90it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.5401, Accuracy: 8514/10000 (85.14%)

Learning Rate = 0.01785 for EPOCH 13
EPOCH:  13

Loss=0.39318379759788513 Batch_id=97 Accuracy=85.96: 100%|██████████| 98/98 [00:16<00:00,  6.04it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.3895 --> 0.3861).  Saving model ...

Test set: Average loss: 0.3861, Accuracy: 8534/10000 (85.34%)

Learning Rate = 0.01639 for EPOCH 14
EPOCH:  14

Loss=0.3470646142959595 Batch_id=97 Accuracy=87.24: 100%|██████████| 98/98 [00:16<00:00,  6.01it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.3861 --> 0.3719).  Saving model ...

Test set: Average loss: 0.3719, Accuracy: 8605/10000 (86.05%)

Learning Rate = 0.01493 for EPOCH 15
EPOCH:  15

Loss=0.3942098617553711 Batch_id=97 Accuracy=88.32: 100%|██████████| 98/98 [00:16<00:00,  5.96it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3995, Accuracy: 8614/10000 (86.14%)

Learning Rate = 0.01347 for EPOCH 16
EPOCH:  16

Loss=0.25187841057777405 Batch_id=97 Accuracy=88.81: 100%|██████████| 98/98 [00:16<00:00,  5.94it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.4002, Accuracy: 8632/10000 (86.32%)

Learning Rate = 0.01201 for EPOCH 17
EPOCH:  17

Loss=0.31911975145339966 Batch_id=97 Accuracy=89.69: 100%|██████████| 98/98 [00:16<00:00,  5.94it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3890, Accuracy: 8702/10000 (87.02%)

Learning Rate = 0.01055 for EPOCH 18
EPOCH:  18

Loss=0.22628377377986908 Batch_id=97 Accuracy=90.39: 100%|██████████| 98/98 [00:16<00:00,  6.01it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.4641, Accuracy: 8716/10000 (87.16%)

Learning Rate = 0.00909 for EPOCH 19
EPOCH:  19

Loss=0.22087712585926056 Batch_id=97 Accuracy=91.12: 100%|██████████| 98/98 [00:16<00:00,  6.00it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.3719 --> 0.3645).  Saving model ...

Test set: Average loss: 0.3645, Accuracy: 8691/10000 (86.91%)

Learning Rate = 0.00763 for EPOCH 20
EPOCH:  20

Loss=0.2925671637058258 Batch_id=97 Accuracy=91.85: 100%|██████████| 98/98 [00:16<00:00,  5.95it/s]
  0%|          | 0/98 [00:00<?, ?it/s]

Validation loss has  decreased (0.3645 --> 0.3058).  Saving model ...

Test set: Average loss: 0.3058, Accuracy: 8827/10000 (88.27%)

Learning Rate = 0.00617 for EPOCH 21
EPOCH:  21

Loss=0.19671748578548431 Batch_id=97 Accuracy=92.23: 100%|██████████| 98/98 [00:16<00:00,  6.01it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3216, Accuracy: 8839/10000 (88.39%)

Learning Rate = 0.00472 for EPOCH 22
EPOCH:  22

Loss=0.1904577910900116 Batch_id=97 Accuracy=92.86: 100%|██████████| 98/98 [00:16<00:00,  5.95it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.4843, Accuracy: 8825/10000 (88.25%)

Learning Rate = 0.00326 for EPOCH 23
EPOCH:  23

Loss=0.21792027354240417 Batch_id=97 Accuracy=93.29: 100%|██████████| 98/98 [00:16<00:00,  6.00it/s]
  0%|          | 0/98 [00:00<?, ?it/s]


Test set: Average loss: 0.3090, Accuracy: 8884/10000 (88.84%)

Learning Rate = 0.0018 for EPOCH 24
EPOCH:  24

Loss=0.1875913292169571 Batch_id=97 Accuracy=93.91: 100%|██████████| 98/98 [00:16<00:00,  6.03it/s]


Test set: Average loss: 0.3944, Accuracy: 8920/10000 (89.20%)

Learning Rate = 0.00034 for EPOCH 25

