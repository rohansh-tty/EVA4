# S8 Assignment

Task: 

    Go through this repository: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
    Extract the ResNet18 model from this repository and add it to your API/repo. 
    Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
    Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
    Once done finish S8-Assignment-Solution


Assignment Solution: ![ResNet Model]()

* Model Features:

1. Used GPU
2. ResNet Variant: ResNet18
3. Total Params: 11,173,962
4. Used L1 & L2 since the model was Overfitting. 
5. Also Trained the model a bit harder by adding few Image Augmentation Techniques like RandomRotation, HorizonatalFlip & Vertical Flip.
6. Ran 20 Epochs with Highest Train Accuracy: 91.84% & Corresponding Test Accuracy: 89.22% 
