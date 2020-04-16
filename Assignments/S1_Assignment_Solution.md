

**1.What are Channels and Kernels (according to EVA)?**
Image may have single or multiple channels. Channel is like a container of similar features or information. It's a bag of features having same characteristics. For eg a Normal(color) image will have 3 channels like Red, Green, Blue and Black-White Image will have a single channel
![](https://blog.xrds.acm.org/wp-content/uploads/2016/06/Figure1.png)

Kernels are feature extractors. There are different types of kernels which have specific uses, like some type of kernels help in detecting vertical edges and some help in detecting horizontal ones.
![](https://www.researchgate.net/profile/Volker_Weinberg/publication/332190148/figure/fig2/AS:743933420249088@1554378957080/Schematic-illustration-of-a-convolutional-operation-The-convolutional-kernel-shifts-over.ppm)
 
**2. Why should we (nearly) always use 3x3 kernels?**
3x3 Kernel
Usually Even shaped kernels lack line of symmetry, for example if a 2x2 kernel is used to detect a vertical edge, it can detect edge but it won't have its other portion or there's no symmetry. And this is why most of kernels used are of odd sized ones i.e 3x3, 5x5 etc. Also 3x3 kernels can act as a base component for large sized kernels.

Serial No 		Kernel Size 	 Number of Parameters
		
1 	3x3 		9
2 	5x5 		25
3 	7x7 		49
4(Instead of 5x5) 	2*(3x3) 	18
5(Instead of 7x7) 	3*(3x3) 	27

 

**3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)?**

```
convsize = 199 #set the layer size to 199
while convsize - 2 > 0:
                 print(str(convsize) + 'x' + str(convsize)+ '>'+ str(convsize - 2)+ 'x'+str(convsize - 2))
                 convsize = convsize - 2   #decrement the layer size by 2
```
 

**4. How are kernels initialized?**

Usually Kernels are randomly initialized. It's not set to zeros, which otherwise would give all input neurons the same weight resulting in same output. Instead Kernels are set to arbitrary values. And later using SGD technique, they are set to optimal values.

 

**5. What happens during the training of a DNN?**

Training of DNN
With respect to Computer Vision, DNNs have numerous applications, like Object Detection, Image Classification, Face Recognition etc. For DNNs the input will be the image dataset related to the problemset. The image dataset is divided into different classes according to the output. DNN used for these image related problems are called Convolutional Neural Network. CNN includes Convolutional Layers which convolve the image with specific filters, to extract certain features. CNN includes Padding and Pooling layers (Links to an external site.). Padding helps in retaining the same image size as that of input whereas Pooling helps in reducing the image to a smaller representation.
