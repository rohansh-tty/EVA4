import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Block

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        # Residual Block
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                stride = stride, padding = 1, bias = False)
        self.bn1 = nn.Sequential(nn.BatchNorm2d(planes))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                                stride = stride, padding = 1, bias=False)
        self.bn2 = nn.Sequential(nn.BatchNorm2d(planes))

        # # Shortcut
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes,
        #                   kernel_size=3,padding=1, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes))
        #         # nn.MaxPool2d((2,2)))
        #     # self.shortcut = nn.Sequential(nn.MaxPool2d((2,2)))
          
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print("shape of out x", out.shape)
        # print("shape of shortcut x", self.shortcut(x).shape)
        out = out + x
        out = F.relu(out)
        return out

# customNet Class
class customNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(customNet,self).__init__()
        self.in_planes = 128

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False))
        self.bn1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU())
                            
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.pool1 = nn.MaxPool2d((2,2))
        self.bn2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU())

        self.res1 = self._make_layer(block, 128, num_blocks[0], stride = 1) # I have assigned num_blocks to 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride =1, padding = 1, bias = False)
        self.pool2 = nn.MaxPool2d((2,2))
        self.bn3 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU())

        self.conv4 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.pool3 = nn.MaxPool2d((2,2))
        self.bn4 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU())

        self.in_planes = 512
        self.res2 = self._make_layer(block, 512, num_blocks[1], stride = 1)
        
        self.pool4 = nn.MaxPool2d((4,4))
        self.linear = nn.Linear(512, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        # print('strides', strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
            # print('self.in_planes', self.in_planes)
        # print("layers", layers)
        return nn.Sequential(*layers)

    
    def forward(self, x):

        """
        Function Variables:
        l1X: X at Layer 1
        l2X: X at Layer 2
        """
        l1X = self.conv1(x)
        l1X = self.bn1(l1X)
        l1X = self.conv2(l1X)
        l1X = self.pool1(l1X)
        l1X = self.bn2(l1X)
        # print("l1X shape", l1X.shape)
       

        res1 = self.res1(l1X)
        # print("res1 shape", res1.shape)
        res1X = res1 + l1X
        

        l2X = self.conv3(res1X)
        l2X = self.pool2(l2X)
        l2X = self.bn3(l2X)

        l3X = self.conv4(l2X)
        l3X = self.pool3(l3X)
        l3X = self.bn4(l3X)

        res2 = self.res2(l3X)
        res2X = res2 + l3X

        outX = self.pool4(res2X)
        outX = outX.view(outX.size(0), -1)
        outX = self.linear(outX)

        return outX


def main11():
  return customNet(BasicBlock, [1,1])



