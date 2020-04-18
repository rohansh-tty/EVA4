## S4 Assignment

**Main Task**


    WRITE MNIST HANDWRITTEN DIGIT CLASSIFIER AGAIN SUCH THAT IT ACHIEVES
        99.4% validation accuracy
        Less than 20k Parameters
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






