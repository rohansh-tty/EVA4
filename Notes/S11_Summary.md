# Super-Convergence

Research Paper:![Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates
Leslie N. Smith, Nicholay Topin](https://arxiv.org/abs/1708.07120)

Abstract: In this paper, we describe a phenomenon, which we called "super-convergence", where neural networks can be trained an order of magnitude faster than with standard training methods. The existence of super-convergence is relevant to understanding why deep networks generalize well.
When you implement Super-Convergence, large learning rates regularize the training process, so you might have to reduce other forms of Regularization in order to maintain the balance. 

* **Learning Rate Annealing**

Selection of the optimal starting learning rate is the important. Because if you start with a lower learning rate then it would take a lot of time to reach minima and if you start with a higher learning rate then that it won't converge very well. 

![](https://miro.medium.com/max/918/0*uIa_Dz3czXO5iWyI.)


After the setting of initial learning rate you have decay it in such a way you won't miss out on minima. 

