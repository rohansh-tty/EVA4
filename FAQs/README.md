# FAQs

Here in this sub-repo I'm gonna share few important things related to Computer Vision. 

![](https://i.pinimg.com/originals/07/40/20/074020eefceeef41b251dd257239ada3.jpg)

1. I've found it surprising that Numpy arrays and PIL images have different shape, in this case, (H,W) in numpy and (W,H) in PIL. This is because Numpy is not an imaging library. ![numpy.ndarray.shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) gives the shape in this order (H, W, D) to stay coherent with the terminology used in ndarray's axis (axis=0, axis=1, axis=2)

2. Why do we have to normalize images before passing it into CNN?

If we don't divide the feature values/pixels by 255 then the range of feature values in the training vectors would be different. 

3. What's an Epoch? And how is it different from Iterations?

During each Epoch, you feed your network with all the examples present in the training dataset. Data is all passed in the form of batches. Once you feed all the examples, the network gets updated. But you'll not be passing the entire data at once, but in the form of mini batches simultaneously. And each turn you pass a batch is called an Iteration. Usually, each epoch will have multiple iterations. 

For eg: You have built a Hot-Not Hot Dog Classifier where you've used 1000 training examples. Now if you set batch size is 100, then you'll have 10 batches(if you divide equally). For testing model, say you've used around 20 epochs. Your network will feed 10 mini batches each consisting of 100 images simultaneously to the network during each epoch. And this is done 20 times. 


