# FAQs

Here in this sub-repo I'm gonna share few important things related to Computer Vision. 

![](https://i.pinimg.com/originals/07/40/20/074020eefceeef41b251dd257239ada3.jpg)

1. I've found it surprising that Numpy arrays and PIL images have different shape, in this case, (H,W) in numpy and (W,H) in PIL. This is because Numpy is not an imaging library. ![numpy.ndarray.shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) gives the shape in this order (H, W, D) to stay coherent with the terminology used in ndarray's axis (axis=0, axis=1, axis=2)

