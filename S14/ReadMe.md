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

    bg: (d) Background images
    fg: (d) Foreground images
    fg_mask: (d) Mask of foreground images
    bg_fg: (d) Images where foregrounds are overlayed on top of backgrounds
    bg_fg_mask: (d) Mask of overlayed foreground-background images
    bg_fg_depth_map: (d) Depth maps of overlayed foreground-background images
    
The dataset is created using only 100 Background and 100 Foreground Images. 




