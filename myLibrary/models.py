from torchsummary import summary
import torch
import torch.nn
import torch.nn.Functional as F

#------------------------------------------------FirstModel------------------------------------------

# class Net(nn.Module):

# 	def __init__(self, model = DefaultModel): # create a default model
# 		super(Net, self).__init__()
# 		self.Training = None
# 		self.model = model

# 	# Now define few helper functions like conv2d, depthwise convolution, summary etc
# 	def conv2d(self, in_channels, out_channels, kernel_size = (3,3), padding=1, dilation=1, groups=1, bias=False):
		
# 		return [nn.conv2d(in_channels = in_channels, out_channels = onut_channels, kernel_size = kernel_size, padding = padding,
# 		 dilation = dilation, groups = groups, bias = bias)]


# 	def depthwise_conv2d(self, in_channels, out_channels, kernel_size = (3,3), padding=1, dilation=1, groups=1, bias = False):
		
# 		return [nn.conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding,
# 		 dilation = dilation, groups = in_channels, bias = bias), 
# 		nn.conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), bias = bias)]

# # method to add BatchNorm, Dropout & ReLU
# 	def bn_relu_drp(self, convlist, out_channels, bn = True, dropout_value = 0, relu = True):
# 		if bn:
# 			convlist.append(nn.BatchNorm2d(out_channels))
# 		if dropout_value > 0:
# 			convlist.append(nn.Dropout(dropout_value))
# 		if relu:
# 			convlist.append(nn.ReLU())

# 		return nn.Sequential(*convlist)	


# 	def create_conv2d(self, in_channels, out_channels, kernel_size = (3,3), padding=0, dilation=1, groups=1, bias=False):
# 		return bn_relu_drp(self.conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding,
# 		 dilation = dilation, groups = groups, bias = bias), out_channels, bn, dropout, relu)

# # depthwise convolution
# 	def create_dwc(self, in_channels, out_channels, kernel_size = (3,3), padding=1, dilation=1, groups=1, bias = False):
# 		return bn_relu_drp(self.depthwise_conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding,
# 		 dilation = dilation, groups = in_channels, bias = bias), out_channels, bn, dropout, relu)


# 	def summary(self, input_size):
# 		summary(self, input_size = input_size)


# # add your different variant models here in the form of class

# class cifarNet(Net):
# 	def __init__(self, name = 'Model1', dropout_value = 0):
# 		super(cifarNet, self).__init__(name)

# 		# ConvBlock 1
# 		self.conv1 = self.create_conv2d(3, 16, padding = 1, dropout_value)
# 		self.conv2 = self.create_conv2d(16, 32, padding = 1, dropout_value)
# 		self.conv3 = self.create_conv2d(32, 64, padding = 1, dilation = 2, dropout_value)
# 		self.conv4 = self.create_conv2d(64, 64, padding = 1, dilation = 2, dropout_value)

# 		# Transition Block 1
# 		self.pool1 = nn.MaxPool2d(2,2)
# 		self.conv5 = self.create_conv2d(64, 16, kernel_size = (1,1), padding = 1, dropout_value)


# 		# Conv Block 2
# 		self.conv6 = self.create_dwc(16, 32, dropout_value)
# 		self.conv7 = self.create_conv2d(32, 64, padding = 1, dilation = 2, dropout_value)


# 		# Transition Block 2
# 		self.pool2 = nn.MaxPool2d(2,2)
# 		self.conv8 = self.create_conv2d(64, 16, kernel_size = (1,1), padding = 1, dropout_value)


# 		# Conv Block 3
# 		self.conv9 = self.create_conv2d(16, 32, padding = 1, dropout_value)


# 		# Transition Block 3
# 		self.pool3 = nn.MaxPool2d(2,2)
# 		self.conv10 = self.create_conv2d(32, 16, kernel_size = (1,1), dropout_value)


# 		# Conv Block 4 
# 		self.conv11 = self.create_conv2d(16, 16, padding = 1)


# 		# GAP
# 		self.gap = nn.AvgPool2d(kernel_size = (3,3))
# 		self.conv12 = self.create_conv2d(16, 10, kernel_size = (1,1), padding = 0, bn = False, relu = False)


# 	def forward(self, x):
# 		x = self.conv1(x)
# 		x = self.conv2(x)
# 		x = self.conv3(x)
# 		x = self.conv4(x)

# 		x = self.pool1(x)

# 		x = self.conv5(x)
# 		x = self.conv6(x)
# 		x = self.conv7(x)

# 		x = self.pool2(x)

# 		x = self.conv8(x)
# 		x = self.conv9(x)

# 		x = self.pool3(x)
		
# 		x = self.conv10(x)
# 		x = self.conv11(x)

# 		x = self.gap(x)

# 		x = self.conv12(x)

# 		x = x.view(-1, 10)
# 		return F.log_softmax(x, dim = -1)


#-----------------------------------------------SEAFARNET----------------------------------------------------------------#

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, input, output, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(input, input, kernel_size=3, padding=padding, groups=input, bias=bias)
        self.pointwise = nn.Conv2d(input, output, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class seaFarNet(nn.Module): # concise model
    def __init__(self, name="Model"):
        super(seaFarNet, self).__init__()
        self.name = name

# Conv Block1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3), padding = 2, bias = False), 
            nn.ReLU(),
            nn.BatchNorm2d(16)) # O/P: 34  RF: 3
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 34  RF: 5

      # Dilated Convolution 1
        self.dilatedconv1 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 32  RF: 9

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = 1 ,  bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128)) # O/P: 32  RF: 11

# Transition Block 1
        self.pool1 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 16  RF: 12
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = (1,1), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 18  RF: 12      
        

# ConvBlock 2
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 18  RF: 16

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = 1, dilation = 2,  bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128)) # O/P: 16  RF: 24

# Transition Block 2  
        self.pool2 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 8  RF: 26s

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = (1,1), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 10  RF: 26


# ConvBlock 3
        self.dwconv1 = nn.Sequential(
            DepthwiseSeparableConv2d(32, 64, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 10  RF: 34

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 10  RF: 42

# Transition Block 3 
        self.pool3 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 5  RF: 46

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1,1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128)) # O/P: 5  RF: 46


# ConvBlock 4
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(256)) # O/P: 3  RF: 62

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1), padding = 0, bias = False))
          # nn.ReLU(),
          # nn.BatchNorm2d(512)) # O/P: 3  RF: 62
       

    # GAP
        self.gap = nn.Sequential(nn.AvgPool2d(3))
    # Last Layer
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels=10, kernel_size=(1,1), padding=0, bias=False)) # O/P:3  RF: 62

    # Dropout Layer
        self.drop = nn.Sequential(nn.Dropout(0.10))




    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.dilatedconv1(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.drop(x)
        x = self.conv6(x)
        x = self.drop(x)

        x = self.pool2(x)
        x = self.conv7(x)
        x = self.drop(x)
        x = self.dwconv1(x)
        x = self.conv8(x)
        x = self.drop(x)

        x = self.pool3(x)
        x = self.conv9(x)
        x = self.drop(x)
        x = self.conv10(x)
        x = self.drop(x)
        x = self.conv11(x) 
        
        x = self.gap(x)
        x = self.conv12(x)


        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# model = seaFarNet()

#----------------------------------------------_CfarResNet---------------------------------

from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.Functional as F

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, input, output, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(input, input, kernel_size=3, padding=padding, groups=input, bias=bias)
        self.pointwise = nn.Conv2d(input, output, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class cfarResNet(nn.Module): 
''' concise model & 
	similar to ResNet interms of channels(maintaining same no of channels in each block)'''

    def __init__(self, name="Model"):
        super(cfarResNet, self).__init__()
        self.name = name


# Conv Block1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3), padding = 2, bias = False), 
            nn.ReLU(),
            nn.BatchNorm2d(16)) # O/P: 34  RF: 3
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 34  RF: 5

      # Dilated Convolution 1
        self.dilatedconv1 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = 1, dilation = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 32  RF: 9

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = 1 ,  bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 32  RF: 11


# Transition Block 1
        self.pool1 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 16  RF: 12
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (1,1), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 18  RF: 12      
        

# ConvBlock 2
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 18  RF: 16

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 1, dilation = 2,  bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 16  RF: 24


# Transition Block 2  
        self.pool2 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 8  RF: 26

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (1,1), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 10  RF: 26


# ConvBlock 3
        self.dwconv1 = nn.Sequential(
            DepthwiseSeparableConv2d(32, 64, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 10  RF: 34

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 10  RF: 42


# Transition Block 3 
        self.pool3 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 5  RF: 46

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1,1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128)) # O/P: 5  RF: 46


# ConvBlock 4
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(128)) # O/P: 3  RF: 62

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (1,1), padding = 0, bias = False))
          # nn.ReLU(),
          # nn.BatchNorm2d(512)) # O/P: 3  RF: 62
       

    # GAP 
        self.gap = nn.Sequential(nn.AvgPool2d(3))
    # Last Layer
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels=10, kernel_size=(1,1), padding=0, bias=False)) # O/P:3  RF: 62


    # Dropout Layer
        self.drop = nn.Sequential(nn.Dropout(0.10))




    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.dilatedconv1(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.drop(x)
        x = self.conv6(x)
        x = self.drop(x)

        x = self.pool2(x)
        x = self.conv7(x)
        x = self.drop(x)
        x = self.dwconv1(x)
        x = self.conv8(x)
        x = self.drop(x)

        x = self.pool3(x)
        x = self.conv9(x)
        x = self.drop(x)
        x = self.conv10(x)
        x = self.drop(x)
        x = self.conv11(x) 
        
        x = self.gap(x)
        x = self.conv12(x)


        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


