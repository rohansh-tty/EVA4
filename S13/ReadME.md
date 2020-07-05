# S13 Assignment

**Yolo 2 & 3**


Assignment Task: 

    OpenCV Yolo: SOURCE (Links to an external site.)
        Run this above code on your laptop or Colab. 
        Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
        Run this image through the code above. 
        Upload the link to GitHub implementation of this
        Upload the annotated image by YOLO. 
    Training Custom Dataset on Colab for YoloV3
        Refer to this Colab File: LINK (Links to an external site.)
        Refer to this GitHub Repo (Links to an external site.)
        Collect a dataset of 500 images and annotate them. Please select a class for which you can find a YouTube video as well. Steps are explained in the readme.md file on GitHub.
        Once done:
            Download (Links to an external site.) a very small (~10-30sec) video from youtube which shows your class. 
            Use ffmpeg (Links to an external site.) to extract frames from the video. 
            Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
            Inter on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
            `python detect.py --conf-thres 0.3 --output output_folder_name`
            Use ffmpeg (Links to an external site.) to convert the files in your output folder to video
            Upload the video to YouTube. 
        Share the link to your GitHub project with the steps as mentioned above
        Share the link of your YouTube video
        Share the link of your YouTube video on LinkedIn, Instagram, etc! You have no idea how much you'd love people complimenting you! 


# ASSIGNMENT B
---------

* **MOTIVATION**

Waste Segregator Robo, which classifies and seperates Recyclable, Non Recyclable Waste Materials and dump them into respective Bins.
For example I have collected/got dataset from few GitHub repos and planning to build this project.

* **IMPLEMENTATION**
1. For S13 B, I have Metal Cans as my class. Dataset includes Metal Cans, Caps, Aluminium Foils etc. Collected around 300-400 Images from this repo, rest from the Internet.
2. Used this for Annotation. Link for the Annotation Tool
3. Ran for 110 Epochs. 


* **SAMPLES**

**1. Train Data**
![](https://github.com/Gilf641/EVA4/blob/master/S13/Images/train_batch0.png)

**2. Test Data**
![Test Samples](https://github.com/Gilf641/EVA4/blob/master/S13/Images/train_batch0.png)




