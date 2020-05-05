# Architectural Basics

* Fully Connected Layers

![](https://pvsmt99345.i.lithium.com/t5/image/serverpage/image-id/42339i8BA3F2CCCEDE7458?v=1.0)

> Each line you see is the weight we are training. Those circles are "temporary" values that will be stored. Once you train the model, lines are what all matter!

1. Temporary in the sense, Circles represent values that are obtained by multiplying input with weights, since the inputs change, multiplying inputs with weights also change. 
2. In broader sense, Circles represent calculated neuron value and these values will change for every input image.
3. Also those lines(called Weights) in the above image are all what matters because we're training the model to find the right value. And they help in representing the density of connections which inturn indicates the type of model


* Why flattening 2D image to 1D(single vector) doesn't make any sense?

![](https://github.com/Gilf641/Test/blob/master/ezgif-6-07ff0eb0db4e.gif)
1. It's stripping away information. Operating a CNN using this kind of data makes learning pretty hard. 
2. Slight shift in object can result in different vector which inturn will predict wrong answer.
3. While looking 2D data we're focusing on Spatial information, after we convert 2D to 1D we've lost spatial meaning. Without Spatial information it would really hard to train a Vision DNN. 


* Why you shouldn't use 2D data for Fully Connected Layer/s?

1. FCN is translationally variant for 2D. 
2. 1D form of data works best for FCN.

* Why not use bias?

From math, this is the equation for straight line
> y = mx + c 

And for this c is useful only when x = 0

This is the equation for multiple regression 
> y = m1x1 + m2x2 + m3x3 + m4x4 + m5x5 + ... + mnxn + c 

(where mn is the nth coeffecient, xn is the indepenedent variable while c is bias)

For this particular equation, c aka bias is useful only when all independent variables are equal to zero which is not possible.

* Power of Convolution

![](https://miro.medium.com/max/1052/0*Asw1tDuRs3wTjwi7.gif)

Let's say for example we've color image of 5x5 now we're convolving with 9 kernels of size 3x3 to get output of channel size 3x3. In this we've around 9 weights(kernels) and total number of connections equals to 81, while if you're doing FCN using 25 pixels and then having 9 neurons in next layer, that will sum up total connections and weight params to 225.

| Type of Layer| Total Connections | Weights |
| ------------- | ------------- | ------------- |
| Convolution Layer | 81  | 9 |
| Fully Connected Layer | 225  | 225 |
