import copy
import os
import torch
from tqdm.autonotebook import tqdm
import torch.optim as optim
import torch.nn as nn 
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class CyclicLR:
    def __init__(self, max_lr, min_lr, stepsize, num_iterations):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.stepsize = stepsize
        self.iterations = num_iterations
        self.lr_list = []

    def cycle(self, iteration):
        return int(1 + (iteration/(2*self.stepsize)))

    def lr_position(self, iteration, cycle):
        return abs(iteration/self.stepsize - (2*cycle) + 1)

    def current_lr(self, lr_position):
        return self.min_lr + (self.max_lr - self.min_lr)*(1-lr_position)
    
    def cyclic_lr(self, plotGraph = True):
        for i in range(self.iterations):
            cycle = self.cycle(i)
            lr_position = self.lr_position(i, cycle)
            current_lr = self.current_lr(lr_position)
            self.lr_list.append(current_lr)
        
        if plotGraph:
            fig = plt.figure(figsize=(12,5))
            
            #Plot Title
            plt.title('Cyclic LR Plot')

            plt.xlabel('Iterations')
            plt.ylabel('Learning Rate')

            plt.axhline(y=self.min_lr, label='min lr', color='r')
            plt.text(0, self.min_lr, 'min lr')

            plt.axhline(y=self.max_lr, label='max lr', color='r')
            plt.text(0, self.max_lr, 'max lr')

            plt.plot(self.lr_list)




LR_List = []
Acc_List = []
def lr_rangetest(device, 
                model,
                trainloader, 
                criterion,  
                minlr, 
                maxlr, 
                epochs,
                weight_decay=0.05,
                pltTest=True):
    """
    Args:-
    """
    lr = minlr

    for e in range(epochs):
        testModel = copy.deepcopy(model)
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=weight_decay)
        lr = lr + (maxlr-minlr)/epochs
        testModel.train()
        pbar = tqdm(trainloader)
        correct, processed = 0, 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = testModel(data)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct = correct + pred.eq(target.view_as(pred)).sum().item()
            processed = processed + len(data)
            print('EPOCH {n}:-'.format(n=e))
            pbar.set_description(desc=f'LR={optimizer.param_groups[0]["lr"]}  Loss={loss.item()}  Batch_id={batch_idx}  Accuracy={100*correct/processed:0.2f}')
        Acc_List.append(100*correct/processed)
        LR_List.append(optimizer.param_groups[0]['lr'])
    
    if pltTest:
        with plt.style.context('fivethirtyeight'):
            plt.plot(LR_List, Acc_List)
            plt.xlabel('Learning Rate')
            plt.ylabel('Accuracy')
            plt.title('Learning Rate Range Test')
            plt.show()

