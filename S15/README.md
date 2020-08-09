# S15 Image Segmentation & Depth Estimation

Inspired by Simon Sinek, I always try to do things by asking 3 Questions, **WHY**, **HOW** & **WHAT**  	:dart:



# **Why do we need Image Segmentation & Depth Estimation?** :man_shrugging:
![](https://github.com/Gilf641/EVA4/blob/master/S15/images/final%20collage.png)

Image Segmentation is applied almost everywhere, Healthcare, Video Surveilliance, Detection & Recognition Tasks etc. Depth Estimation is used in Augmented Reality, Robotics & Object Trajectory estimation. 

Okay so why am I doing it?/Motivation *Two Reasons...*

First,

* This is EVA's Capstone and Phase 2 Qualifying project.
    
And second

* I'm here to work in Deep Learning Field as long as I can. And I believe, understanding the end-to-end process/cycle of what goes inside any project is important. This project is one of a kind. 
And there's a lot to learn, from **creating a large dataset of 1.2 Million Images**, later creating a stable **data pipeline to transform images** to customizing a neural net in a such a way that, on passing **a background and a background+foreground image as an input**, it is capable of doing multi-tasking i.e **predicting depth and do image segmentation.**



# **What is Image Segmentation & Depth Estimation?**

Image Segmentation means classifying every pixel(objects in whole)  present in an image based on it's class and Depth Estimation is process of estimating object depth available in a scene/image. 

# **How to do Image Segmentation & Depth Estimation?**

## **Contents**

1. ![Problem Statement](https://github.com/Gilf641/EVA4/tree/master/S15#problem-statement)
2. ![Dataset Creation](https://github.com/Gilf641/EVA4/tree/master/S15#dataset-creation)
3. ![Data Processing](https://github.com/Gilf641/EVA4/tree/master/S15#data-processing)
4. ![Choosing a good DNN Architecture](https://github.com/Gilf641/EVA4/tree/master/S15#choosing-a-good-dnn-architecture)
5. ![Model Evaluation](https://github.com/Gilf641/EVA4/tree/master/S15#model-evaluation)
6. ![Displaying Test Results](https://github.com/Gilf641/EVA4/tree/master/S15#displaying-test-results)
7. ![Implementation](https://github.com/Gilf641/EVA4/tree/master/S15#implementation)
8. ![References](https://github.com/Gilf641/EVA4/tree/master/S15#references)


## Problem Statement
    Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. 


## Dataset Creation 
 The dataset consists of 6 parts,

    BG: Background Images
    FG: Foreground Images
    FG MASK: Mask of Foreground Images
    BG_FG: Images of Foregrounds overlayed on Backgrounds
    BG_FG MASK: Mask of overlayed Foreground-Background images
    BG_FG DEPTH MAP: Depth maps of overlayed Foreground-Background images
    
Total no of Images in this dataset is 1.2 Million. The dataset is created using only 100 Background and 100 Foreground Images. 

Dataset Description:

*For detailed explanation, read ![this](https://github.com/Gilf641/EVA4/tree/master/S14)*

**1. BG**

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/BG.png)

This directory contains office images. These are my background images for this particular dataset.

    Image Size: 224x224x3
    Number of Images: 100
    Naming Convention: bg001.jpeg, bg002.jpeg, ..., bg100.jpeg



**2. FG**

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/FG.png)

This directory contains transparent foreground images, in which you can see people walking, talking, posing together etc.
Now for this set of images, I have applied Horizontal Flip. So the total count of FG Images is 200.

    Image Size: 224x224x3
    Number of Images: 200
    Naming Convention: fg001.png, fg001_01.png, fg002.png, fg002_01.png

*fg001_01.png indicates Horizontal Flip of Original Image fg001.png*

**3. FG MASK**

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/FG%20MASK.png)

This directory consists of Foreground Masks. 

    Image Height: 110
    Number of Channels: 1
    Number of Images: 200
    Naming Convention: frg001.png, frg001_01.png, frg002.png, frg002_01.png



**4. BG_FG**

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/FGBG.png)

This directory contains images where random foregrounds are overlayed on different backgrounds at **n** random positions. These are called background-foreground images in short BG_FG.

    N: 20
    Image Size: 224x224x3
    Number of Images: 400,000
    Naming Convention: bg001_fg001_001.jpeg, bg001_fg001_002.jpeg
        
*bg001_fg001_001.jpeg means that fg001 was overlayed on bg001 at the first random position.*

**5. BG_FG MASK**

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/FGBG%20MASK.png)

This directory contains mask of BG_FG images. These are called background-foreground masks in short BG_FG Masks.

    Image Size: 224x224x1
    Number of Images: 400,000
    Naming Convention: I haven't decided yet!

**6. BG_FG DEPTH MAP**

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/DP%20MAP.png)

This directory contains depth map of BG_FG images generated from Dense Depth Model. These are called background-foreground depth maps.

    Image Size: 224x224x1
    Number of Images: 400,000
    Naming Convention: bg_001_fg_001_001_depth_map_001.jpeg, bg_001_fg_001_002_depth_map_002.jpeg

*bg_001_fg_001_001_depth_map_001 indicates the depth map of bg_001_fg_001_001.*



## Data Processing

I planned to work on Single Channel Images, instead of RGB. And I have converted them to Grayscale. Why I did this? Easier to process, with each image having size of 2KB.

It's hard to work on Colab specially if you have large image dataset, in my case 1.2 Million so I have wrapped them up into small batches of 50K*4 which means it'll have 200K images per set. 

Steps:

1. Create a zip file having 50K images of BG, BGFG, BGFG Masks & DepthMaps. So in total the file should consist of 200K images. I prefer adding them in seperate subfolders.
2. Upload it to drive, and Extract the same, using Python's ZipFile library.
3. Copy the path of each folder in drive and assign them to 4 variables. This makes it easy to access the image files inside that folder.
4. Now zip indiviual files using FOR loop.

Common Augmentation across all 4 ImageTypes, Resize with Mandatory Normalization & ToTensor conversion. 


## Choosing a good DNN Architecture

* First Attempt

I tried ResNet34 architecture to solve this problem. I knew it wouldn't work that well, but just wanted to understand how the results were and how it performed.
The results were really bad and also to many model params, and I would often get OOM error on Colab. 

* Second Attempt

So I knew I had to change and started exploring various model architectures used for Image Segmentation. The most common one was UNet, basically it was an Encoder-Decoder Architecture, where Encoder would help in extracting Image Contect while Decoder will do the Localization part for you. And I picked up the Plain & Simple U-Net Architecture.



        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv2d-1         [-1, 16, 192, 192]             880
               BatchNorm2d-2         [-1, 16, 192, 192]              32
                   Dropout-3         [-1, 16, 192, 192]               0
                      ReLU-4         [-1, 16, 192, 192]               0
                    Conv2d-5         [-1, 16, 192, 192]           2,320
               BatchNorm2d-6         [-1, 16, 192, 192]              32
                   Dropout-7         [-1, 16, 192, 192]               0
                      ReLU-8         [-1, 16, 192, 192]               0
                DoubleConv-9         [-1, 16, 192, 192]               0
                MaxPool2d-10           [-1, 16, 96, 96]               0
                   Conv2d-11           [-1, 32, 96, 96]           4,640
              BatchNorm2d-12           [-1, 32, 96, 96]              64
                  Dropout-13           [-1, 32, 96, 96]               0
                     ReLU-14           [-1, 32, 96, 96]               0
                   Conv2d-15           [-1, 32, 96, 96]           9,248
              BatchNorm2d-16           [-1, 32, 96, 96]              64
                  Dropout-17           [-1, 32, 96, 96]               0
                     ReLU-18           [-1, 32, 96, 96]               0
               DoubleConv-19           [-1, 32, 96, 96]               0
               DownSample-20           [-1, 32, 96, 96]               0
                MaxPool2d-21           [-1, 32, 48, 48]               0
                   Conv2d-22           [-1, 64, 48, 48]          18,496
              BatchNorm2d-23           [-1, 64, 48, 48]             128
                  Dropout-24           [-1, 64, 48, 48]               0
                     ReLU-25           [-1, 64, 48, 48]               0
                   Conv2d-26           [-1, 64, 48, 48]          36,928
              BatchNorm2d-27           [-1, 64, 48, 48]             128
                  Dropout-28           [-1, 64, 48, 48]               0
                     ReLU-29           [-1, 64, 48, 48]               0
               DoubleConv-30           [-1, 64, 48, 48]               0
               DownSample-31           [-1, 64, 48, 48]               0
                MaxPool2d-32           [-1, 64, 24, 24]               0
                   Conv2d-33          [-1, 128, 24, 24]          73,856
              BatchNorm2d-34          [-1, 128, 24, 24]             256
                  Dropout-35          [-1, 128, 24, 24]               0
                     ReLU-36          [-1, 128, 24, 24]               0
                   Conv2d-37          [-1, 128, 24, 24]         147,584
              BatchNorm2d-38          [-1, 128, 24, 24]             256
                  Dropout-39          [-1, 128, 24, 24]               0
                     ReLU-40          [-1, 128, 24, 24]               0
               DoubleConv-41          [-1, 128, 24, 24]               0
               DownSample-42          [-1, 128, 24, 24]               0
                MaxPool2d-43          [-1, 128, 12, 12]               0
                   Conv2d-44          [-1, 128, 12, 12]         147,584
              BatchNorm2d-45          [-1, 128, 12, 12]             256
                  Dropout-46          [-1, 128, 12, 12]               0
                     ReLU-47          [-1, 128, 12, 12]               0
                   Conv2d-48          [-1, 128, 12, 12]         147,584
              BatchNorm2d-49          [-1, 128, 12, 12]             256
                  Dropout-50          [-1, 128, 12, 12]               0
                     ReLU-51          [-1, 128, 12, 12]               0
               DoubleConv-52          [-1, 128, 12, 12]               0
               DownSample-53          [-1, 128, 12, 12]               0
                 Upsample-54          [-1, 128, 24, 24]               0
                   Conv2d-55          [-1, 128, 24, 24]         295,040
              BatchNorm2d-56          [-1, 128, 24, 24]             256
                  Dropout-57          [-1, 128, 24, 24]               0
                     ReLU-58          [-1, 128, 24, 24]               0
                   Conv2d-59           [-1, 64, 24, 24]          73,792
              BatchNorm2d-60           [-1, 64, 24, 24]             128
                  Dropout-61           [-1, 64, 24, 24]               0
                     ReLU-62           [-1, 64, 24, 24]               0
               DoubleConv-63           [-1, 64, 24, 24]               0
                 UpSample-64           [-1, 64, 24, 24]               0
                 Upsample-65           [-1, 64, 48, 48]               0
                   Conv2d-66           [-1, 64, 48, 48]          73,792
              BatchNorm2d-67           [-1, 64, 48, 48]             128
                  Dropout-68           [-1, 64, 48, 48]               0
                     ReLU-69           [-1, 64, 48, 48]               0
                   Conv2d-70           [-1, 32, 48, 48]          18,464
              BatchNorm2d-71           [-1, 32, 48, 48]              64
                  Dropout-72           [-1, 32, 48, 48]               0
                     ReLU-73           [-1, 32, 48, 48]               0
               DoubleConv-74           [-1, 32, 48, 48]               0
                 UpSample-75           [-1, 32, 48, 48]               0
                 Upsample-76           [-1, 32, 96, 96]               0
                   Conv2d-77           [-1, 32, 96, 96]          18,464
              BatchNorm2d-78           [-1, 32, 96, 96]              64
                  Dropout-79           [-1, 32, 96, 96]               0
                     ReLU-80           [-1, 32, 96, 96]               0
                   Conv2d-81           [-1, 16, 96, 96]           4,624
              BatchNorm2d-82           [-1, 16, 96, 96]              32
                  Dropout-83           [-1, 16, 96, 96]               0
                     ReLU-84           [-1, 16, 96, 96]               0
               DoubleConv-85           [-1, 16, 96, 96]               0
                 UpSample-86           [-1, 16, 96, 96]               0
                 Upsample-87         [-1, 16, 192, 192]               0
                   Conv2d-88         [-1, 16, 192, 192]           4,624
              BatchNorm2d-89         [-1, 16, 192, 192]              32
                  Dropout-90         [-1, 16, 192, 192]               0
                     ReLU-91         [-1, 16, 192, 192]               0
                   Conv2d-92         [-1, 16, 192, 192]           2,320
              BatchNorm2d-93         [-1, 16, 192, 192]              32
                  Dropout-94         [-1, 16, 192, 192]               0
                     ReLU-95         [-1, 16, 192, 192]               0
               DoubleConv-96         [-1, 16, 192, 192]               0
                 UpSample-97         [-1, 16, 192, 192]               0
                   Conv2d-98          [-1, 1, 192, 192]              17
                  OutConv-99          [-1, 1, 192, 192]               0
               MaxPool2d-100           [-1, 16, 96, 96]               0
                  Conv2d-101           [-1, 32, 96, 96]           4,640
             BatchNorm2d-102           [-1, 32, 96, 96]              64
                 Dropout-103           [-1, 32, 96, 96]               0
                    ReLU-104           [-1, 32, 96, 96]               0
                  Conv2d-105           [-1, 32, 96, 96]           9,248
             BatchNorm2d-106           [-1, 32, 96, 96]              64
                 Dropout-107           [-1, 32, 96, 96]               0
                    ReLU-108           [-1, 32, 96, 96]               0
              DoubleConv-109           [-1, 32, 96, 96]               0
              DownSample-110           [-1, 32, 96, 96]               0
               MaxPool2d-111           [-1, 32, 48, 48]               0
                  Conv2d-112           [-1, 64, 48, 48]          18,496
             BatchNorm2d-113           [-1, 64, 48, 48]             128
                 Dropout-114           [-1, 64, 48, 48]               0
                    ReLU-115           [-1, 64, 48, 48]               0
                  Conv2d-116           [-1, 64, 48, 48]          36,928
             BatchNorm2d-117           [-1, 64, 48, 48]             128
                 Dropout-118           [-1, 64, 48, 48]               0
                    ReLU-119           [-1, 64, 48, 48]               0
              DoubleConv-120           [-1, 64, 48, 48]               0
              DownSample-121           [-1, 64, 48, 48]               0
               MaxPool2d-122           [-1, 64, 24, 24]               0
                  Conv2d-123          [-1, 128, 24, 24]          73,856
             BatchNorm2d-124          [-1, 128, 24, 24]             256
                 Dropout-125          [-1, 128, 24, 24]               0
                    ReLU-126          [-1, 128, 24, 24]               0
                  Conv2d-127          [-1, 128, 24, 24]         147,584
             BatchNorm2d-128          [-1, 128, 24, 24]             256
                 Dropout-129          [-1, 128, 24, 24]               0
                    ReLU-130          [-1, 128, 24, 24]               0
              DoubleConv-131          [-1, 128, 24, 24]               0
              DownSample-132          [-1, 128, 24, 24]               0
               MaxPool2d-133          [-1, 128, 12, 12]               0
                  Conv2d-134          [-1, 128, 12, 12]         147,584
             BatchNorm2d-135          [-1, 128, 12, 12]             256
                 Dropout-136          [-1, 128, 12, 12]               0
                    ReLU-137          [-1, 128, 12, 12]               0
                  Conv2d-138          [-1, 128, 12, 12]         147,584
             BatchNorm2d-139          [-1, 128, 12, 12]             256
                 Dropout-140          [-1, 128, 12, 12]               0
                    ReLU-141          [-1, 128, 12, 12]               0
              DoubleConv-142          [-1, 128, 12, 12]               0
              DownSample-143          [-1, 128, 12, 12]               0
                Upsample-144          [-1, 128, 24, 24]               0
                  Conv2d-145          [-1, 128, 24, 24]         295,040
             BatchNorm2d-146          [-1, 128, 24, 24]             256
                 Dropout-147          [-1, 128, 24, 24]               0
                    ReLU-148          [-1, 128, 24, 24]               0
                  Conv2d-149           [-1, 64, 24, 24]          73,792
             BatchNorm2d-150           [-1, 64, 24, 24]             128
                 Dropout-151           [-1, 64, 24, 24]               0
                    ReLU-152           [-1, 64, 24, 24]               0
              DoubleConv-153           [-1, 64, 24, 24]               0
                UpSample-154           [-1, 64, 24, 24]               0
                Upsample-155           [-1, 64, 48, 48]               0
                  Conv2d-156           [-1, 64, 48, 48]          73,792
             BatchNorm2d-157           [-1, 64, 48, 48]             128
                 Dropout-158           [-1, 64, 48, 48]               0
                    ReLU-159           [-1, 64, 48, 48]               0
                  Conv2d-160           [-1, 32, 48, 48]          18,464
             BatchNorm2d-161           [-1, 32, 48, 48]              64
                 Dropout-162           [-1, 32, 48, 48]               0
                    ReLU-163           [-1, 32, 48, 48]               0
              DoubleConv-164           [-1, 32, 48, 48]               0
                UpSample-165           [-1, 32, 48, 48]               0
                Upsample-166           [-1, 32, 96, 96]               0
                  Conv2d-167           [-1, 32, 96, 96]          18,464
             BatchNorm2d-168           [-1, 32, 96, 96]              64
                 Dropout-169           [-1, 32, 96, 96]               0
                    ReLU-170           [-1, 32, 96, 96]               0
                  Conv2d-171           [-1, 16, 96, 96]           4,624
             BatchNorm2d-172           [-1, 16, 96, 96]              32
                 Dropout-173           [-1, 16, 96, 96]               0
                    ReLU-174           [-1, 16, 96, 96]               0
              DoubleConv-175           [-1, 16, 96, 96]               0
                UpSample-176           [-1, 16, 96, 96]               0
                Upsample-177         [-1, 16, 192, 192]               0
                  Conv2d-178         [-1, 16, 192, 192]           4,624
             BatchNorm2d-179         [-1, 16, 192, 192]              32
                 Dropout-180         [-1, 16, 192, 192]               0
                    ReLU-181         [-1, 16, 192, 192]               0
                  Conv2d-182         [-1, 16, 192, 192]           2,320
             BatchNorm2d-183         [-1, 16, 192, 192]              32
                 Dropout-184         [-1, 16, 192, 192]               0
                    ReLU-185         [-1, 16, 192, 192]               0
              DoubleConv-186         [-1, 16, 192, 192]               0
                UpSample-187         [-1, 16, 192, 192]               0
                  Conv2d-188          [-1, 1, 192, 192]              17
                 OutConv-189          [-1, 1, 192, 192]               0
        ================================================================
        Total params: 2,161,666
        Trainable params: 2,161,666
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.84
        Forward/backward pass size (MB): 289.41
        Params size (MB): 8.25
        Estimated Total Size (MB): 298.50
        ----------------------------------------------------------------



 The first priority that I had in mind was 'let-me-build-a-bad-model-first-later-customize-it'. This I had to do  to understand the nuances of this problem and also how the model behaved.

The inputs and the outputs are not of same resolution. Inputs are of size 192x192 while the Outputs are 96x96. The model has an encoder-decoder architecture, where the model takes two inputs: BG and BG-FG and returns two outputs: Depth Map and Mask. The inputs are first individually processed through one encoder block each and then fed to a same network.



Training
I had to be careful while working with COLAB, automatic runtime disconnection, OOM error etc. So I planned to train my model on small size datasets. Dataset size was b/w 30K-40K and I ran for 15-20 Epochs. I used this ![file]() for training my model.

Short notes on Model Training

    The model was trained on smaller resolution images first and then gradually the image resolution was increased.
    BCE and IoU were used as evaluation metrics. BCE was calculated on mask outputs while IoU was calculated on depth outputs.
    Reduce LR on Plateau with patience of 2 and threshold 1e-3.
    Auto model checkpointing which saved the model weights after every epoch.


## Model Evaluation

This is the hardest part. I tried multiple things together, which made the Model Stats look horrendous.  Atlast I made my mind to use something basic BCEWithLogitsLoss for Mask & IoU loss for Depth Maps along with SSIM. I added SSIM because, it's an improv for MSE, which only focuses on Pixel Degradation as change in Structural Info, whereas MSE considers Mean Absolute Errors. Also I noticed SSIM working like a charm for Depth Estimation. 


![](https://github.com/Gilf641/EVA4/blob/master/S15/images/max_ssim.gif)

I couldn't properly evaluate my model and might need some more time to finish. Not sure what's wrong, asked everyone in my group to take a look at my code, debugged it myself for over a week or something, still my network is predicting blank masks, but depth estimation is average. 


## Implementation 

**MASK PREDICTION & DEPTH ESTIMATION**

    Parameters - 2,161,666
    Optimiser - Adam/SGD
    Scheduler - Reduce Lr On Pleateau
    Loss - BCE, IoU with SSIM
    Total No Of Epochs trained for - 10
    Total No of data used - 10k

## Displaying Test Results


## References


