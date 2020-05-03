# S6 Assignment
Task:
 Your assignment 6 is:

    * take your 5th code
    * run your model for 25 epochs for each:
        without L1/L2 with BN
        without L1/L2 with GBN
        with L1 with BN
        with L1 with GBN
        with L2 with BN
        with L2 with GBN
        with L1 and L2 with BN
        with L1 and L2 with GBN
    *You cannot be running your code 8 times manually (-500 points for that). You need to be smarter and write a single loop or iterator to iterate through these conditions. 
    *draw ONE graph to show the validation accuracy curves for all 8 jobs above. This graph must have proper legends and it should be clear what we are looking at. 
    *draw ONE graph to show the loss change curves for all 8 jobs above. This graph must have proper legends and it should be clear what we are looking at. 
    *find any 25 misclassified images for "without L1/L2 with BN" AND "without L1/L2 with GBN" model. You should be using the saved model from the above jobs. 
    and L2 models. You MUST show the actual and predicted class names.

I have modified my 5th Assignment code & tuned the model to work on above mentioned conditions. Since I was frequently getting this *Buffered data was truncated after reaching the output size limit* error it forced me to run my iterator twice and also reduce the epochs to 24(only for GBN part).

**![S6 Assignment Solution](https://github.com/Gilf641/EVA4/blob/master/S6/S6_Assignment_Solution.ipynb)**


## Data Visualization

* Validation Accuracy for all 8 jobs above
![](https://github.com/Gilf641/EVA4/blob/master/S6/validationacc_BN_GBN.png)


* Validation Loss for all 8 jobs above
![](https://github.com/Gilf641/EVA4/blob/master/S6/validationloss_BN_GBN.png)


* 25 misclassified images for without L1/L2 with BN
![](https://github.com/Gilf641/EVA4/blob/master/S6/withoutL1_L2_BN(1).png)


* 25 misclassified images for without L1/L2 with GBN
![](https://github.com/Gilf641/EVA4/blob/master/S6/withoutL1_L2_GBN(1).png)


* Explain your observation w.r.t. L1 and L2's performance in the regularization of your model.

1. I have taken both L1 factor & weight_decay(for L2) as 1e-5. 
2. In the BN model with L1 regularization I saw a dip in validation accuracy at end. Whereas for L2 regularization, there was a spike in accuracy which it maintained till the end of 24 epochs. 
3. In terms of Validation Loss, one can notice the fluctuation in the curve for L1 model, while for L2 it was true in the initial epochs. 

