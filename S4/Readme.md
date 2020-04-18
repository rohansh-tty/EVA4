## S4 Assignment

 We have considered many many points in our last 4 lectures. Some of these we have covered directly and some indirectly. They are:

    How many layers,
    MaxPooling,
    1x1 Convolutions,
    3x3 Convolutions,
    Receptive Field,
    SoftMax,
    Learning Rate,
    Kernels and how do we decide the number of kernels?
    Batch Normalization,
    Image Normalization,
    Position of MaxPooling,
    Concept of Transition Layers,
    Position of Transition Layer,
    DropOut
    When do we introduce DropOut, or when do we know we have some overfitting
    The distance of MaxPooling from Prediction,
    The distance of Batch Normalization from Prediction,
    When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
    How do we know our network is not going well, comparatively, very early
    Batch Size, and effects of batch size
    etc (you can add more if we missed it here)
    
    
    
**Main Task**


    WRITE MNIST HANDWRITTEN DIGIT CLASSIFIER AGAIN SUCH THAT IT ACHIEVES
        99.4% validation accuracy
        Less than 20k Parameters
        You can use anything from above you want. 
        Less than 20 Epochs
        No fully connected layer
        To learn how to add different things we covered in this session, you can refer to this code: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 (Links to an external site.) DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.


###### Assignment Solution: ![S4-Solution](https://github.com/Gilf641/EVA4/blob/master/S4/S4-Assignment-Solution.ipynb)


* Model Keypoints:
1. 3 Convolution Blocks
2. BatchNormalization after each layer
3. Dropout at the end of each block
4. AvgPool instead of MaxPool
5. 17,594 Parameters
5. Global Average Pooling at last






