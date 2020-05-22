**Kernels, Activation & Layers**

CNNs follow the hierarchy strictly like first Edges & Gradients are detected, then something complicated like Textures & Patterns are formed as a result of conv operation/s conducted on Edges by Kernels, then parts of object and lastly the entire object

![](https://developer.nvidia.com/sites/default/files/pictures/2018/convolutional_neural_network.png)

* But the problem arises when you try to decrease the number of channels after MaxPooling Layers. And this can be solved using 3x3 Kernel. But it's not effecient since there's sudden surge in number of params after MaxPool Layer. 


| Input  | Kernel | Output | Receptive Field |
| ------------- | ------------- | ------------- | ------------- |
400x400x3     | (3x3x3)x32        | 398x398x32   |  RF of 3x3
398x398x32   | (3x3x32)x64      | 396x396x64    |RF of 5X5
396x396x64   | (3x3x64)x128    | 394x394x128  |RF of 7X7
394x394x128 | (3x3x128)x256 | 392x392x256  |RF of 9X9
392x392x256 | (3x3x256)x512 | 390x390x512  |RF of 11X11
MaxPooling
195x195x512 | **(?x?x512)x32**   | **?x?x32** | RF of 22x22
.. 3x3x32x64| (3x3x32)x64   |   |RF of 24x24
.. 3x3x64x128| (3x3x64)x128   |  |RF of 26x26
.. 3x3x128x256| (3x3x128)x256  | |RF of 38x28
.. 3x3x256x512 |  (3x3x256)x512   |  |RF of 30x30


In this case we implement **1x1 Kernel** instead of 3x3


![](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif)

But the reason behind using this is important. There are 3 different reasons why it's replaced.

1. Number of parameters is computationally less

This is one of the main reason it's implemented. If you use 3x3 kernel after MaxPooling layer, then number of parameters is 9 times more than what we get when convoluted with 1x1 Kernel. 
![Difference b/w using 1x1 & 3x3 Kernel](https://i.stack.imgur.com/4ki2u.png)


2. Computationally Faster

Since the number of parameters is less when convoluted using 1x1 kernel, the computation process becomes fast. 


3. Easy and Simple Computation

When an image gets convoluted with 1x1 kernel, what's actually happening is that a particular value of 1x1 kernel gets multiplied to each and every pixel of the input image. And this is a simple multiplication. So 1x1 convolution makes process easy and simple




**Features of 1x1 Convolution**

*1x1 Kernel helps in filtering features.
*1x1 is computation less expensive. 
*1x1 Convolution is similar to multiplying the whole channel with just 1 number here it's pixel value.



