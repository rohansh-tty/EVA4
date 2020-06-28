# Object Detection & Localization

* **Introduction**

Image Classification is a Computer Vision task where the model has to classify the objects present in the input image. While for Object Detection, the model has locate the object and mark it down either using Bounding Box or Mask. 


**Logits** are the most underrated feature in Computer Vision. No one uses this directly, either they pass it directly to Cross Entropy Loss or convert it Softmax value and then later pass it onto Negative Likelihood Loss. Now we need to understand what these Logits actually convey. They talk about the amount of input features that model can extract, if the logit is low, it means less number of features are present and vice-versa. Logits usually depend o  the Input-Image features, Quantity of Features that are extractable, Intensity of Features. Now to this we can add some threshold and filter only significant ones, which inturn can improve model performance.

It's not necessary that in an image, all objects are of same size or atleast let me put it this way. In a dataset, even though size of image is same, object size may vary. Now for the model to perform better it should be change its Receptive Field or in simple words, the model should have multiple Receptive Fields. This will improve its performance over such kind of datasets. Example: VGG has constant RF, like of fixed range, now if you pass any image having object of size beyond or below this range than VGG would fail to predict the correct class, if you're doing ImageClassification.

* **Note:**

Classification is also a part of Object Detection. We first do the image classification, understand what's there in the image, pass the classification output which is Logits to the Object Detection network. 

* Two Important Object Detection Algorithms
1. YOLO Like Approach
2. SSD Like Approach

One major difference b/w these two is that SSD uses Predefined Template/Bounding Boxes, whereas YOLO doesn't! 

* Bounding Box
Bounding Boxes are good, but they have some serious problem.
1. Initial size of Bounding Box. How will you fix this? 
2. Now suppose you chose the right Bounding Box(not exactly but almost), once detected in which direction will you expand it to make it almost as if it fits that object.
3. Again assuming you chose the right box, but you have multiple objects in the input image of different size, how will you resize BBox for all of them?

Sounds Impossible right!...

There's another method called Region Proposal. A complex method but at depth solves most of the problems caused by BBox. Used in MaskRCNN. Some of steps include
1. Divide the whole image based on similar components.
2. Gather similar/like features.
3. Pass these as an input to a CNN.


* Anchor Box
Named by YOLO Team, invented by SSD Developers, Anchor Boxes are Template boxes of different sizes which you along with Batch to a DNN. While in SSD Algo, the template box sizes are predefined, whereas in YOLO, the model predicts Sx and Sy which alter the Template size. 


One approach for Object Detection would be to divide the input image to NxN Blocks and find in which block the Object centroid lies in. N would depend on the Dataset. The output neuron in YOLO model would look something like this. 
(N x N x Number of Templates x Number of Classes)

SSD's Predefined Anchor Boxes are of 1:1, 3:4 & 4:3 scale. 

* How to calculate Anchor Boxes? 
Link to Excel File

* Intersection Over Union(IoU)

IoU implies ratio of Intersected Area over the whole Union/Total Area.
For ith Image 
IoU(i) = min(width ,cx[i])* min(height , cy[i])/(height x width + cx[i] x cy[i]- min(width , cx[i]) x min(height ,cy[i]))

where cx[i], cy[i] are the centroid coordinates, while width and height are the normalised metrics of BBox. 

