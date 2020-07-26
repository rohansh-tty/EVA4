# Monocular Depth Estimation Office Dataset Creation

* **What is Monocular Depth Estimation?**

The problem  we're trying to solve is to estimate the depth of an object using single image, instead of an image sequence or image pair. Therefore the name Monocular Depth Estimation 

*And why can't we use the latter one, reason due to unavailability of good sized dataset.*

* **Some traditional geometry-based methods for Depth Estimation**

1. Stereo Vision Matching
2. Structure from Motion

*They can be used to calculate depth values of sparse points. But these methods are heavily dependent on the image pairs or image sequences.*


# Directory Structure

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




# Data Statistics

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
    



