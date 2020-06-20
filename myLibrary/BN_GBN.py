# Python Package to load Batch Normalization & Ghost Batch Normalization


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
import mnist_execute as execute


# BN Model
class Net(nn.Module):
    def __init__(self, GBN=False):
      super(Net, self).__init__()

      # Conv Block1 
      self.convblock1 = nn.Sequential(
          nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), padding = 0, bias = False), 
          nn.ReLU(),
          nn.BatchNorm2d(8)) # O/P: 26
      
      # Conv Block2
      self.convblock2 = nn.Sequential(
          nn.Conv2d(in_channels = 8, out_channels = 10, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          nn.BatchNorm2d(10)) # O/P: 24

      # MaxPool Layer
      self.maxpool = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 12

      # ConvBlock 3
      self.convblock3 = nn.Sequential(
          nn.Conv2d(in_channels = 10, out_channels = 16, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          nn.BatchNorm2d(16)) # O/P: 10
      
      
      # ConvBlock 4
      self.convblock4 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          nn.BatchNorm2d(16)) # O/P: 8
      
      # ConvBlock 5
      self.convblock5 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          nn.BatchNorm2d(16)) # O/P: 6

      # ConvBlock 6
      self.convblock6 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 0,  bias = False))
         

    # GAP
      self.gap = nn.Sequential(nn.AvgPool2d(4))

    
    # Last Layer
      self.convblock9 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = (1,1), padding = 0, bias = False))


    # Dropout Layer
      self.drop = nn.Sequential(nn.Dropout(0.08))



    def forward(self,x):
      x = self.convblock1(x)
      x = self.drop(x)
      x = self.convblock2(x)
      x = self.drop(x)
      x = self.maxpool(x)
      x = self.convblock3(x)
      x = self.drop(x)
      x = self.convblock4(x)
      x = self.drop(x)
      x = self.convblock5(x)
      x = self.convblock6(x)
      x = self.gap(x)
      x = self.convblock9(x)
     
      x = x.view(-1, 10)
      return F.log_softmax(x, dim = -1)
      

# Different Version of Batch Normalized Model i.e without L1 & L2, with L1, with L2, with L1 & L2

class BN_Models(Net):
  def __init__(self, model, device, trainloader, testloader, epochs=25):
    super(BN_Models, self).__init__()
    self.model = model
    self.epochs = epochs
    self.device = device
    self.trainloader = trainloader
    self.testloader = testloader
    self.acc = [] # this is to plot Validation Accuracy Curve
    self.loss = [] # this is to plot Validation Loss Curve


# WITHOUT L1 & L2 REGULARIZATION + BATCH-NORM
  def withoutL1_L2_BN(self):
    model = Net().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
    print('without L1 and L2 regularization with BN')
    model1= execute.Test_Train()
    global loss1
    global acc1
    loss1 = model1.test_losses
    acc1 = model1.test_acc
    self.acc.append(acc1)
    self.loss.append(loss1)

    for epoch in range(1,self.epochs+1):
      print("EPOCH:", epoch)
      model1.train(model, self.device, self.trainloader, optimizer, epoch, scheduler)
      model1.test(model, self.device, self.testloader,"model1.pt")



# WITH L1 REGULARIZATION + BATCH-NORM

  def withL1_BN(self):
    model =  Net().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
    #second model
    print('with L1 regularization with BN')
    model2 = execute.Test_Train()
    global loss2
    global acc2
    loss2 = model2.test_losses
    acc2 = model2.test_acc
    self.acc.append(acc2)
    self.loss.append(loss2)

    for epoch in range(1,self.epochs+1):
        print("EPOCH:", epoch)
        model2.train(model, self.device, self.trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
        model2.test(model, self.device, self.testloader,"model2.pt")

# WITH L2 REGULARIZATION + BATCH-NORM
  def withL2_BN(self):
    model =  Net().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
    #third model
    print('with L2 regularization with BN')
    model3 = execute.Test_Train()
    global loss3
    global acc3
    loss3 = model3.test_losses
    acc3 = model3.test_acc
    self.acc.append(acc3)
    self.loss.append(loss3)

    for epoch in range(1,self.epochs+1):
        print("EPOCH:", epoch)
        model3.train(model, self.device, self.trainloader, optimizer, epoch, scheduler)
        model3.test(model, self.device, self.testloader,"model3.pt")



# WITH L1 & L2 REGULARIZATION + BATCH-NORM
  def withL1_L2_BN(self):
    model =  Net().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
  #fourth  model
    print('with L1 & L2 regularization with BN')
    model4 = execute.Test_Train()
    global loss4
    global acc4
    loss4 = model4.test_losses
    acc4 = model4.test_acc
    self.acc.append(acc4)
    self.loss.append(loss4)

    for epoch in range(1,self.epochs+1):
        print("EPOCH:", epoch)
        model4.train(model, self.device, self.trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
        model4.test(model, self.device, self.testloader,"model4.pt")


# TO PLOT VALIDATION ACCURACY CURVE
  def bn_plot_acc(self, figname):
    self.figname = figname
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.plot(self.acc[0])
    ax.plot(self.acc[1])
    ax.plot(self.acc[2])
    ax.plot(self.acc[3])
    ax.set(title="Validation Accuracy of 4 Models BatchNormalization", xlabel="Epoch", ylabel="Accuracy")
    ax.legend(['without L1 and L2', 'with L1', 'with L2', 'with L1 and L2'], loc='best')
    plt.savefig(str(self.figname)+'.png')
    plt.show()


# TO PLOT VALIDATION LOSS CURVE
  def bn_plot_loss(self, figname):
    self.figname = figname
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.plot(self.loss[0])
    ax.plot(self.loss[1])
    ax.plot(self.loss[2])
    ax.plot(self.loss[3])
    ax.set(title="Validation Loss of 4 Models BatchNormalization", xlabel="Epoch", ylabel="Loss")
    ax.legend(['without L1 and L2', 'with L1', 'with L2', 'with L1 and L2'], loc='best')
    plt.savefig(str(self.figname)+'.png')
    plt.show()










# GBN MODEL
class GBNet(nn.Module):
    def __init__(self):
      super(GBNet, self).__init__()

      # Conv Block1 
      self.convblock1 = nn.Sequential(
          nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), padding = 0, bias = False), 
          nn.ReLU(),
          GhostBatchNorm(8,2)) # O/P: 26
      
      # Conv Block2
      self.convblock2 = nn.Sequential(
          nn.Conv2d(in_channels = 8, out_channels = 10, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          GhostBatchNorm(10,2)) # O/P: 24

      # MaxPool Layer
      self.maxpool = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 12

      # ConvBlock 3
      self.convblock3 = nn.Sequential(
          nn.Conv2d(in_channels = 10, out_channels = 16, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          GhostBatchNorm(16,2)) # O/P: 10
      
      
      # ConvBlock 4
      self.convblock4 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          GhostBatchNorm(16,2)) # O/P: 8
      
      # ConvBlock 5
      self.convblock5 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 0, bias = False),
          nn.ReLU(),
          GhostBatchNorm(16,2)) # O/P: 6

      # ConvBlock 6
      self.convblock6 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = 0,  bias = False))
         

    # GAP
      self.gap = nn.Sequential(nn.AvgPool2d(4))

    
    # Last Layer
      self.convblock9 = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = (1,1), padding = 0, bias = False))


    # Dropout Layer
      self.drop = nn.Sequential(nn.Dropout(0.08))



    def forward(self,x):
      x = self.convblock1(x)
      x = self.drop(x)
      x = self.convblock2(x)
      x = self.drop(x)
      x = self.maxpool(x)
      x = self.convblock3(x)
      x = self.drop(x)
      x = self.convblock4(x)
      x = self.drop(x)
      x = self.convblock5(x)
      # x = self.drop(x)
      x = self.convblock6(x)
      x = self.gap(x)
      x = self.convblock9(x)
     
      x = x.view(-1, 10)
      return F.log_softmax(x, dim = -1)
      


# Different Version of Ghost Batch Normalized Model i.e without L1 & L2, with L1, with L2, with L1 & L2

class GBN_Models(GBNet):
  def __init__(self, device, trainloader, testloader, epochs=25):
    super(GBN_Models, self).__init__()
    self.epochs = epochs
    self.device = device
    self.trainloader = trainloader
    self.testloader = testloader
    self.acc = []
    self.loss = []

# WITHOUT L1 & L2 REGULARIZATION + GBN
  def withoutL1_L2_GBN(self):
    model = GBNet().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
    #fifth model
    print('without L1 and L2 regularization with GBN')
    model5 = execute.Test_Train()
    global loss5
    global acc5
    loss5 = model5.test_losses
    acc5 = model5.test_acc
    self.acc.append(acc5)
    self.loss.append(loss5)

    for epoch in range(1,self.epochs+1):
      print("EPOCH:", epoch)
      model5.train(model, self.device, self.trainloader, optimizer, epoch, scheduler)
      model5.test(model, self.device, self.testloader,"model5.pt")


# WITH L1 REGULARIZATION + GBN
  def withL1_GBN(self):
    model =  GBNet().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
    #sixth model
    print('with L1 regularization with GBN')
    model6 = execute.Test_Train()
    global loss6
    global acc6
    loss6 = model6.test_losses
    acc6 = model6.test_acc
    self.acc.append(acc6)
    self.loss.append(loss6)

    for epoch in range(1,self.epochs+1):
        print("EPOCH:", epoch)
        model6.train(model, self.device, self.trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
        model6.test(model, self.device, self.testloader,"model6.pt")



# WITH L2 REGULARIZATION + GBN
  def withL2_GBN(self):
    model =  GBNet().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
    #seventh model
    print('with L2 regularization with GBN')
    model7 = execute.Test_Train()
    global loss7
    global acc7
    loss7 = model7.test_losses
    acc7 = model7.test_acc
    self.acc.append(acc7)
    self.loss.append(loss7)

    for epoch in range(1,self.epochs+1):
        print("EPOCH:", epoch)
        model7.train(model, self.device, self.trainloader, optimizer, epoch, scheduler)
        model7.test(model, self.device, self.testloader,"model7.pt")


# WITH L1 & L2 REGULARIZATION + GBN
  def withL1_L2_GBN(self):
    model =  GBNet().to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(self.trainloader), epochs=25)
  #fourth  model
    print('with L1 & L2 regularization with GBN')
    model8 = execute.Test_Train()
    global loss8
    global acc8
    loss8 = model8.test_losses
    acc8 = model8.test_acc
    self.acc.append(acc8)
    self.loss.append(loss8)

    for epoch in range(1,self.epochs+1):
        print("EPOCH:", epoch)
        model8.train(model, self.device, self.trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
        model8.test(model, self.device, self.testloader,"model8.pt")



  def gbn_plot_acc(self, figname):
    self.figname = figname
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.plot(self.acc[0])
    ax.plot(self.acc[1])
    ax.plot(self.acc[2])
    ax.plot(self.acc[3])
    ax.set(title="Validation Accuracy of 4 Models Ghost-BatchNormalization", xlabel="Epoch", ylabel="Accuracy")
    ax.legend(['without L1 and L2', 'with L1', 'with L2', 'with L1 and L2'], loc='best')
    plt.savefig(str(self.figname)+'.png')
    plt.show()

  def gbn_plot_loss(self, figname):
    self.figname = figname
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.plot(self.loss[0])
    ax.plot(self.loss[1])
    ax.plot(self.loss[2])
    ax.plot(self.loss[3])
    ax.set(title="Validation Loss of 4 Models Ghost-BatchNormalization", xlabel="Epoch", ylabel="Loss")
    ax.legend(['without L1 and   L2', 'with L1', 'with L2', 'with L1 and L2'], loc='best')
    plt.savefig(str(self.figname)+'.png')
    plt.show()



# Ghost Batch Normalization

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0) # fill_() helps you filling up the tensor with a particular data
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))
    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


      