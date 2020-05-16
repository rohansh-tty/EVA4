# S8 Assignment

Task: 

    Go through this repository: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
    Extract the ResNet18 model from this repository and add it to your API/repo. 
    Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
    Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
    Once done finish S8-Assignment-Solution


**Assignment Solution**: ![ResNet Model](https://github.com/Gilf641/EVA4/blob/master/S8/S8_AssignmentSolution.ipynb)

## **Model Features:**

1. Used GPU
2. ResNet Variant: ResNet18
3. Total Params: 11,173,962
4. Since the model was Overfitting I used L1 & L2
5. Also Trained the model a bit harder by adding few Image Augmentation Techniques like RandomRotation, HorizonatalFlip & Vertical Flip. Didn't make the mistake of adding all transformations together, but experimented with the first one, analysed the  model performance, later added second and lastly included the third one. 
6. Used CrossEntropyLoss() to calculate loss value.
7. Ran the model for 20 Epochs with 

        * Highest Train Accuracy: 91.84% 

        * Corresponding Test Accuracy: 89.22% 

* **Model Analysis:**
1. Lot of fluctuations in Validation Loss values. 
2. Not that Overfit Model
3. In Misclassified Images, one can see that most of images are either hidden / occluded / oriented in different way. Also in some images the class deciding portions is kinda dark. Eg: AirPlane Image (2nd Row, 4th Column) with it's wings, rear parts are not that visible. Front portion of Truck( 5th row, 2nd column)is excluded.






https://github.com/Gilf641/EVA4/blob/master/S8/S8_AssignmentSolution.ipynb
## **Library Documentation:**

1.![image_transformations.py](https://github.com/Gilf641/EVA4/blob/master/S8/evaLibrary/image_transformations.py) : Applies required image transformation to both Test & Train dataset aka Image PreProcessing.

2.![resNet.py](https://github.com/Gilf641/EVA4/blob/master/S8/evaLibrary/resNet.py): Consists of 2 models i.e seafarNet & cfarResNet(don't mind the names...)

3.![execute.py](https://github.com/Gilf641/EVA4/blob/master/S8/evaLibrary/execute.py): Scripts to Test & Train the model.

4.![DataLoaders.py](https://github.com/Gilf641/EVA4/blob/master/S8/evaLibrary/DataLoaders.py): Scripts to load the dataloaders.

5.![visualizeData.py](https://github.com/Gilf641/EVA4/blob/master/S8/evaLibrary/visualizeData.py): Consists of helper functions to plot images from dataset & misclassified images

6.![rohan_library](https://github.com/Gilf641/EVA4/blob/master/S8/evaLibrary/rohan_library.py): Imports all the required libraries at once.


## **Misclassified Images**

![](https://github.com/Gilf641/EVA4/blob/master/S8/Misclassified%20Ones.png)
