# EVA4 - TSAI
*This repo is related to EVA Program(Extensive Vision AI) offered by The School of AI, Bangalore. Consists of Assignments(& Solutions), Notes, Related Projects etc.*


18 Sessions spanned over 15 weeks, 2 hours per session is what will make Phase 1. This would be covering tons of research papers and their implementations i.e InceptionNet, ResNet, DenseNet, EfficientNet etc.

Till now 4 sessions are completed. I will briefly summarise what I have learned and share some insights here. Star this Repo and Stay Tuned for upcoming sessions.


![](https://media.giphy.com/media/Ln2dAW9oycjgmTpjX9/giphy.gif)


* Session 1: Background & Basics

 Image may have single or multiple channels. Channel is similar to a container of **like** features or information.
 For eg: a normal image will have 3 channels like Red, Green, Blue.
![](https://pointeradvertising.com/wp-content/uploads/2013/06/RGB-CMYK-When-Where.gif)


Kernels are feature extractors. There are different types of kernels which have specific uses, like some type of kernels help in detecting vertical edges, while some help in detecting horizontal ones etc.
For eg: The Kernel below is a Horizontal Edge Detector
![](https://miro.medium.com/max/3146/1*EDqq5ZHYyJE70Zvdt1K_vA.png)

**Importance of 3x3 Kernel**
Usually the Even sized kernels lack line of symmetry, for example if a 2x2 kernel is used to detect a vertical edge, it can detect edge but it won't have its other portion or there's no symmetry. And this is why most of kernels used are of odd sized ones i.e 3x3, 5x5 etc. Also 3x3 kernels can act as a base component for large sized kernels.

Usually Kernels are randomly initialized. It's not set to zeros, which otherwise would give all input neurons the same weight resulting in same output. Instead Kernels are set to arbitrary values. And later using SGD technique, they are set to optimal values.

**What is a CNN?**
CNN stands for Convolutional Neural Network, belongs to the class of Deep Neural Networks which is used to analyze visual imagery.

*Hierarchy of Feature Detection by CNN*
1. Edges & Gradients
2. Textures and Patterns
3. Parts of Object
4. Object Identifiers


* Session 2: Neural Network Concepts & Pytorch 101 for Vision
* Session 3: Kernels, Activations and Layers
* Session 4: CNN Architectural Basics
