## Neural Architecture

* **Neural Nets**

Neural Nets also called as Artificial Neural Network(ANN) are computing systems that can learn on its own. NNs generally perform supervised learning tasks, building knowledge from data sets where the right answer is provided in advance. The networks then learn by tuning themselves to find the right answer on their own, increasing the accuracy of their predictions.


![Simple Neural Network](https://miro.medium.com/max/1063/0*u-AnjlGU9IxM5_Ju.png)



* **Convolution Operation**

According to ![Wikipedia](https://en.wikipedia.org/wiki/Convolution), In mathematics (in particular, functional analysis), Convolution is a mathematical operation on two functions (f and g)that produces a third function expressing how the shape of one is modified by the other.

In CNN, Convolution is a image related operation between an image( later a convoluted one) and a Kernel that produces a convoluted image with shape calculated using the below formula.

![](https://miro.medium.com/max/660/0*_r70kZaBlXSyZzz5.)



* **Why do we add more layers?** 
 
In initial layers of CNN, it detects small features like **Edges and Gradients**. They then *make something complex out of it* and this loop goes on. This also depends on the task at hand. 

![MNIST Data](https://external-preview.redd.it/Dhrpp8M4X9BpyFOKGpD6uxl2aFRC3fBS-akgcZ2cxYw.gif?format=png8&s=8a9143099e235e11018e7adcadcb7b7973f5e4c1)


* **Intuition behind CNN Hierarchy**
Here I'm gonna relate CNN Hierarchy with *Words/Sentence* analogy

 1. Edges & Gradients   <----->  Letters
 2. Textures & Patterns <----->  Words
 3. Parts of Object     <----->  Sentences/Phrases
 4. Object Identifiers  <----->  Paragraphs
 5. Complete Object     <----->  Pages
 
 
 * **GPUs**
 
 ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQKZSnL0Wmsbq4THsaOHxIxyOa0FYulCN5t3UUwNWfVrIWNqdQf&usqp=CAU)
 

 *Since we're gonna use GPUs for running heavy models, we might need to understand how they actually work?*
 
 GPUs usually work on Parallel Computing Concept. Modern GPUs have around 1000 cores and they've organized 16 or 32 SIMD(Single Instruction Multiple Data) blocks. Say for eg if you want to compute 1+1, you will be allotted with 32 cores to compute this thing. Later at the end 31 results will be discarded.  
 
GPUs are really bad at doing one thing at a time. I like to think of it as a **multitasker**.

**CPU v/s GPU**

Think of it this way: If you want to operate a single instruction on multiple pieces of data in this case GPUs are better than CPU. And if you want to do multiple things on one piece of data, CPUs are far better.


* More detail on Convolution
The values you get as a result of convolution is simply put the confidence of the feature existing in that particular image. And you can amplify a particular feature by multiplying a positive number while you deamplify others. 

![Convolution Operation](https://miro.medium.com/max/3412/1*xBkRA7cVyXGHIrtngV3qlg.png)
