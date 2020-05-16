# Batch Normalization & Regularization

S5 was more of Coding than Concepts. So I will add few theory concepts learned from S5 in S6.


Let's start with


![](https://blog.aryatra.com/wp-content/uploads/2017/02/Problem-Solving-Albert-Einstein-Quotes.jpg)

* **Receptive Field Calculation**


Receptive Field usually determines the frequency of the information. RF of a layer depends upon the Stride used in previous layers and the Kernel Size. 


# Main Concept of Receptive Field

![](https://miro.medium.com/max/2188/1*YpXrr8bN5XyqOlztKPHvDw@2x.png)


1. The receptive field of a neuron in one of the lower layers will represent only a small portion of the image, while the receptive field of a neuron in subsequent higher layers involves a combination of receptive fields from several neurons. 
2. So in general a neuron in a higher layer has a wider perspective and will have look at a larger portion of the image, while a neuron from a lower layer will have a restricted small perspective and will represent a small portion of the original image. 
3. In this way, each successive layer is capable of learning increasingly abstract features of the original image. The use of receptive fields in this fashion is thought to give CNNs an advantage in recognizing visual patterns when compared to other types of neural networks.
