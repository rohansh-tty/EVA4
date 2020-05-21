# Batch Normalization Models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from BN_GBN import * 



# WITHOUT L1 & L2 

def withoutL1_L2_BN():
  model =  Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#first model
  print('without L1 and L2 regularization with BN')

  model1= Test_Train()
  global loss1
  global acc1
  loss1 = model1.test_losses
  acc1 = model1.test_acc

  EPOCHS = 25 
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model1.train(model, device, trainloader, optimizer, epoch, scheduler)
      model1.test(model, device, testloader,"model1.pt")


# WITH L1

def withL1_BN():
  model =  Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#second model
  print('with L1 regularization with BN')

  model2 = Test_Train()
  global loss2
  global acc2
  loss2 = model2.test_losses
  acc2 = model2.test_acc

  EPOCHS = 25
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model2.train(model, device, trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
      model2.test(model, device, testloader,"model2.pt")


# WITH L2

def withL2_BN():
  model =  Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#third model
  print('with L2 regularization with BN')

  model3 = Test_Train()
  global loss3
  global acc3
  loss3 = model3.test_losses
  acc3 = model3.test_acc

  EPOCHS = 25
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model3.train(model, device, trainloader, optimizer, epoch, scheduler)
      model3.test(model, device, testloader,"model3.pt")


# WITH L1 & L2 

def withL1_L2_BN():
  model =  Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#fourth  model
  print('with L1 & L2 regularization with BN')

  model4 = Test_Train()
  global loss4
  global acc4
  loss4 = model4.test_losses
  acc4 = model4.test_acc

  EPOCHS = 25
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model4.train(model, device, trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
      model4.test(model, device, testloader,"model4.pt")


# GHOST BN MODELS

# WITHOUT L1 & L2

def withoutL1_L2_GBN():
  model =  GBNet().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#first model
  print('without L1 and L2 regularization with GBN')
#without L1 and L2 regularization with GBN

  model5= Test_Train()
  global loss5
  global acc5
  loss5 = model5.test_losses
  acc5 = model5.test_acc

  EPOCHS = 25  
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model5.train(model, device, trainloader, optimizer, epoch, scheduler)
      model5.test(model, device, testloader,"model5.pt")


# WITH L1

def withL1_GBN():
  model =  GBNet().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#sixth model
  print('with L1 regularization with GBN')
#with L1 regularization with BN
  model6 = Test_Train()
  global loss6
  global acc6
  loss6 = model6.test_losses
  acc6 = model6.test_acc

  EPOCHS = 25
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model6.train(model, device, trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
      model6.test(model, device, testloader,"model6.pt")


# WITH L2

def withL2_GBN():
  model =  GBNet().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#seventh model
  print('with L2 regularization with GBN')
#with L2 regularization 
  model7 = Test_Train()
  global loss7
  global acc7
  loss7 = model7.test_losses
  acc7 = model7.test_acc

  EPOCHS = 25
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model7.train(model, device, trainloader, optimizer, epoch, scheduler)
      model7.test(model, device, testloader,"model7.pt")



# WITH L1 & L2

def withL1_L2_GBN():
  model =  GBNet().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=25)

#eighth  model
  print('with L1 & L2 regularization with GBN')
#with both L1 & L2 regularization with BN
  model8 = Test_Train()
  global loss8
  global acc8
  loss8 = model8.test_losses
  acc8 = model8.test_acc

  EPOCHS = 25
  for epoch in range(1,EPOCHS+1):
      print("EPOCH:", epoch)
      model8.train(model, device, trainloader, optimizer, epoch, scheduler, L1lambda = 1e-5)
      model8.test(model, device, testloader,"model8.pt")