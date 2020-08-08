# S15 Image Segmentation & Depth Estimation

Inspired by Simon Sinek, I always try to do things by asking 3 Questions, **WHY**, **HOW** & **WHAT**. 


# **Why do we need Image Segmentation & Depth Estimation?**

Image Segmentation is applied almost everywhere, Healthcare, Video Surveilliance, Detection & Recognition Tasks etc. Depth Estimation is used in Augmented Reality, Robotics & Object Trajectory estimation. 

Okay so why am I doing it? *Two Reasons...*

First,

* This is EVA's Capstone and Phase 2 Qualifying project.
    
And second

* I'm here to work in Deep Learning Field as long as I can. And I believe, understanding the end-to-end process/cycle of what goes inside any project is important. This project is one of a kind. 
And there's a lot to learn, from **creating a large dataset of 1.2 Million Images**, later creating a stable **data pipeline to transform images** to customizing a neural net in a such a way that, on passing **a background and a background+foreground image as an input**, it is capable of doing multi-tasking i.e **predicting depth and do image segmentation.**



# **What is Image Segmentation & Depth Estimation?**

Image Segmentation means classifying every pixel(objects in whole)  present in an image based on it's class and Depth Estimation is process of estimating object depth available in a scene/image. 

# **How to do Image Segmentation & Depth Estimation?**

## **Contents**

1. ![Problem Statement]()
2. ![Dataset Creation]()
3. ![Data Augmentation]()
4. ![Data Processing]()
5. ![Model Initialization]() 
6. ![Model Training]()
7. ![Choosing Loss Functions]()
8. ![Evaluating Model Performance]()
9. ![Displaying Test Results]()

## Foreword 

    Designing Models require discipline
    Every step you take must have a purpose
    Trying too many things without order or without any notes is useless


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

**1. BG**

This directory contains office images. These are my background images for this particular dataset.

    Image Size: 224x224x3
    Number of Images: 100
    Naming Convention: bg001.jpeg, bg002.jpeg, ..., bg100.jpeg



**2. FG**

This directory contains transparent foreground images, in which you can see people walking, talking, posing together etc.
Now for this set of images, I have applied Horizontal Flip. So the total count of FG Images is 200.

    Image Size: 224x224x3
    Number of Images: 200
    Naming Convention: fg001.png, fg001_01.png, fg002.png, fg002_01.png

*fg001_01.png indicates Horizontal Flip of Original Image fg001.png*

**3. FG MASK**

This directory consists of Foreground Masks. 

    Image Height: 110
    Number of Channels: 1
    Number of Images: 200
    Naming Convention: frg001.png, frg001_01.png, frg002.png, frg002_01.png



**4. BG_FG**

This directory contains images where random foregrounds are overlayed on different backgrounds at **n** random positions. These are called background-foreground images in short BG_FG.

    N: 20
    Image Size: 224x224x3
    Number of Images: 400,000
    Naming Convention: bg001_fg001_001.jpeg, bg001_fg001_002.jpeg
        
*bg001_fg001_001.jpeg means that fg001 was overlayed on bg001 at the first random position.*

**5. BG_FG MASK**

This directory contains mask of BG_FG images. These are called background-foreground masks in short BG_FG Masks.

    Image Size: 224x224x1
    Number of Images: 400,000
    Naming Convention: I haven't decided yet!

**6. BG_FG DEPTH MAP**

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

First approach to solve this problem Encoder-Decoder Architecture. Like Encoder would help in extracting Image Context while Decoder will do the localization part for you(with the help of cropped background concatenation). So I picked up the plain and simple U-Net Architecture.

 The first priority that I had in mind was 'let-me-build-a-bad-model-first-later-customize-it'. This I had to do  to understand the nuances of this problem and also how the model behaved.


The inputs and the outputs are not of same resolution. Inputs are of size 192x192 while the Outputs are 96x96. The model has an encoder-decoder architecture, where the model takes two inputs: BG and BG-FG and returns two outputs: Depth Map and Mask. The inputs are first individually processed through one encoder block each and then fed to a same network.



## Choosing Loss Functions



## Evaluating Model Performance


## Displaying Test Results


## References