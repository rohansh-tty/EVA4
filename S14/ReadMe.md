# Monocular Depth Estimation Office Dataset Creation


* **Contents** 

1. ![What is Monocular Depth Estimation?](https://github.com/Gilf641/EVA4/tree/master/S14#What-is-Monocular-Depth-Estimation?)
2. ![Directory Structure](https://github.com/Gilf641/EVA4/tree/master/S14#directory-structure)
3. ![Dataset Description](https://github.com/Gilf641/EVA4/tree/master/S14#dataset-description)
4. ![Data Stats](https://github.com/Gilf641/EVA4/tree/master/S14#data-statistics)
5. ![Dataset Creation Part](https://github.com/Gilf641/EVA4/tree/master/S14#dataset-creation-part)

    i. ![Download BG & FG Images](https://github.com/Gilf641/EVA4/tree/master/S14#download-bg--fg-images)
    
    ii. ![Creating Transparent Backgrounds for Foreground Images](https://github.com/Gilf641/EVA4/tree/master/S14#creating-foregrounds-with-transparent-background)
    
    iii. ![Foreground Mask Creation](https://github.com/Gilf641/EVA4/tree/master/S14#foreground-mask-creation)
    
    iv. ![Overlaying Foregrounds on Backgrounds](https://github.com/Gilf641/EVA4/tree/master/S14#overlaying-foregrounds-on-backgrounds)
    
    v. ![Generate Depth Maps](https://github.com/Gilf641/EVA4/tree/master/S14#generate-depth-maps)

## **What is Monocular Depth Estimation?**

The problem  we're trying to solve is to estimate the depth of an object using single image, instead of an image sequence or image pair. Therefore the name Monocular Depth Estimation 

*And why can't we use the latter one, reason due to unavailability of good sized dataset.*


![](https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1000%2F1*8Dtmfv9cbYB84QXwPh9mow.png&imgrefurl=https%3A%2F%2Fmc.ai%2Fdepth-estimation-using-encoder-decoder-networks-and-self-supervised-learning%2F&tbnid=etGc7Ji2lr7pNM&vet=12ahUKEwjT6a3IqerqAhXV_3MBHYeCAacQMygLegUIARC_AQ..i&docid=RwgSIKR9aFyYCM&w=1000&h=311&q=depth%20estimation&client=firefox-b-d&ved=2ahUKEwjT6a3IqerqAhXV_3MBHYeCAacQMygLegUIARC_AQ)


* **Some traditional geometry-based methods for Depth Estimation**

1. Stereo Vision Matching
2. Structure from Motion

*They can be used to calculate depth values of sparse points. But these methods are heavily dependent on the image pairs or image sequences.*


## Problem Statement

## Directory Structure

The dataset consists of 6 parts,

    BG: Background Images
    FG: Foreground Images
    FG MASK: Mask of Foreground Images
    BG_FG: Images of Foregrounds overlayed on Backgrounds
    BG_FG MASK: Mask of overlayed Foreground-Background images
    BG_FG DEPTH MAP: Depth maps of overlayed Foreground-Background images
    
Total no of Images in this dataset is 1.2 Million. The dataset is created using only 100 Background and 100 Foreground Images. 

## Dataset Description:

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

    Image Size: 224x224x3
    Number of Images: 400,000
    Naming Convention: bg_001_fg_001_001_depth_map_001.jpeg, bg_001_fg_001_002_depth_map_002.jpeg

*bg_001_fg_001_001_depth_map_001 indicates the depth map of bg_001_fg_001_001.*




## Data Statistics

Dataset Size:  GB

Number of Images: 1,200,100

Image Types and their Statistics

    Backgrounds(BG)
        Image Size: 224x224x3
        Number of Images: 100
        Mean: (0.5177, 0.5439, 0.5617)
        Standard Deviation: (0.2344, 0.2286, 0.229)

    Background-Foregrounds(BG_FG)
        Image Size: 224x224x3
        Number of Images: 400,000
        Mean: (0.513, 0.536, 0.5546)
        Standard Deviation: (0.2355, 0.2317, 0.2326)

    Background-Foreground Masks(BG_FG MASKS)
        Image Size: 224x224x1
        Number of Images: 400,000
        Mean: 0.05207
        Standard Deviation: 0.21686

    Background-Foreground Depth Maps(DEPTH MAPS)
        Image Size: 224x224x3
        Number of Images: 400,000
        Mean: (0.304, 0.304, 0.304)
        Standard Deviation: (0.1026, 0.1026, 0.1026)
    

# Dataset Creation Part

## Download BG & FG images

    Create the directories BG and FG in the main directory.
    Download 100 Office Images. Crop these images in an aspect ratio of 1:1 and maintain image size 224x224. Save these images in the BG directory.
    Download 100 images containing humans/people walking, standing etc. Save these images in the FG directory. Now it's advised to download images with solid background which makes it easy to remove the background.

Sample background images

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/bg2.png)

## Creating Foregrounds with Transparent Background

Inorder to overlay Foreground Images randomly on top of Background Images, former one must have a transparent background.

For removing backgrounds, I used the open-source software GIMP - GNU Image Manipulation Program. After overlaying, the resulting image should be exported as PNG, because it conserves transparency. Steps for removing background using GIMP has been shown below:



## Foreground Mask Creation

**WHAT'S AN ALPHA CHANNEL?**


   An alpha channel in the in foreground images which indicates the degree of transparency. After adding transparent backgrounds to these images in GIMP, the alpha parameter ranges from 0 (fully transparent) to 255 (fully opaque).
    The alpha channel in foreground images has pixel value set to 0 wherever transparency is present.
    After adding transparency to images in GIMP, the background color of the image is set to white (i.e. pixels values in RGB channel are equal to 255) which is hidden with the help of the alpha channel.

Creating mask

        Mask is created in such a way that the pixels where in the object is present are set to white, while the rest non-object part are set to black.The pixels in the foreground image are set to 255 (white) where the object is present and rest of the pixels (background) are set to 0 (black).
    

**Sample foreground masks**


![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/fgmask2.png)

## Overlaying Foregrounds on Backgrounds

Random Foregrounds are overlayed on different Backgrounds at random positions. So for one Background Image, there'll be 200 Foregrounds and 20 random positions at which these Foregrounds will be overlayed on it. At last, one background you'll have 4000 variants, and for 100 backgrounds, there'll be a total of 400000 BG_FG Images formed.


**Sample BG_FG**


![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/bgfg_2.png)


## Generate Depth Maps

To create these depth map of the BG_FG images, I used ![DenseDepth Model](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb). Implementation for the model inference was referenced from this repository.

Since the depth maps are in grayscale, the number of channels for these images can be reduced to 1, but I haven't done that part yet.

**Sample Depth Maps**

![](https://github.com/Gilf641/EVA4/blob/master/S14/Images/dpmap2.png)

Note: Since we don't have a DepthCam, I relied on a pretrained DenseDepth model to generate depth maps.
