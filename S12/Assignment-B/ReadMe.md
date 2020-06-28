
**Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work)**

File which helps in storing Data Structures & Objects in JavaScript Object Notation is called JSON File. It consists of key-value pairs similar to Python Dictionaries. 
Now in this JSON file there are around 4 Keys/Attributes i.e Filename, Size, Regions and Attributes

Example: 
> "dog11.jpg5671":{"filename":"dog11.jpg","size":5671,"regions":[{"shape_attributes":{"name":"rect","x":97,"y":29,"width":82,"height":74},"region_attributes":{"Class":"Dog"}}],"file_attributes":{"caption":"","public_domain":"no","image_url":""}}

1. First, the Image name which is same as original image name with size attached at the end. This is the main Key. 
2. For this we have around 1 Value, which a dictionary consisting of 4 Keys i.e Filename, Size, Regions and File Attributes.
3. **Filename** is the Original Image Name.
4. **Size** is the image size.
5. **Region** consists of two attributes i.e **Shape** and **Region**
> **Shape Attribute** consists of 4 elements, which refer to the Bounding Box dimension. It consists of **X, Y, W** & ** 
    **(X,Y)** is the starting left (point/corner)coordinate of the bounding box, while W & H are width and height of the Bounding Box. Adding X to W & Y to H results in Bottom Right Coordinate of the Bounding Box. 
    
> While **Region Attribute** consists of **Class** as one of its key, which is equal to the assigned class for that particular object. 
6. Lastly, the **File Attributes** consists of 3 different attributes viz *Caption, Public_domain, image_url*. Caption is related any text passed based on the image. Image_url is link to the particular image. 


**K-Means Clustering**

* Data Distribution Scatter Plot
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/BBX-Data%20Distributio.png)



* Using Elbow Method, best K is 2
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/Elbow%20method.png)


* Calculating IOU for K using this 
>min(width ,cx[i])* min(height , cy[i])/(height x width+ cx[i] x cy[i]- min(wid , cx[i])x min(hei ,cy[i]))
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-A/Images/IOU%20Over%20K.png)


* Clustering Bounding Boxes with K=2
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/K-means%202.png)


* Clustering Bounding Boxes with K=3
![](https://github.com/Gilf641/EVA4/blob/master/S12/Assignment-B/Images/K-means%203.png)

