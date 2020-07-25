import torch
from torch import nn
import torch.nn.functsional as F
import torch.optim as optim


# Double Conv
def double_conv(in_ch, out_ch):
    conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3),
            nn.ReLU(inplace=True))
    return conv


# Crop Image
def CropIt(input_tensor, target_tensor):
    target_size = target_tensor.size()[2]
    input_size = input_tensor.size()[2]
    delta = input_size - target_size # assuming input tensor to be greater than target_size
    delta = delta//2
    return input_tensor[:,:, delta:input_size-delta, delta:input_size-delta]



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)   
        self.down_conv_1 = double_conv(3, 64) 
        self.down_conv_2 = double_conv(64, 128) 
        self.down_conv_3 = double_conv(128, 256) 
        self.down_conv_4 = double_conv(256, 512) 
        self.down_conv_5 = double_conv(512, 1024) 

        self.up_samp_1 = nn.ConvTranspose2d(in_channels=1024,
                                            out_channels=512,
                                            kernel_size=2,
                                            stride=2)

        self.up_conv_1 = double_conv(1024, 512)


        self.up_samp_2 = nn.ConvTranspose2d(in_channels=512,
                                            out_channels=256,
                                            kernel_size=2,
                                            stride=2)

        self.up_conv_2 = double_conv(512, 256)


        self.up_samp_3 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=128,
                                            kernel_size=2,
                                            stride=2)

        self.up_conv_3 = double_conv(256, 128)


        self.up_samp_4 = nn.ConvTranspose2d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=2,
                                            stride=2)

        self.up_conv_4 = double_conv(128, 64)   


        self.out = nn.Conv2d(in_channels=64,
                            out_channels=2,
                            kernel_size=1
        )    
      





    def forward(self, image):
        # encoder
        y1 = self.down_conv_1(image) # CCN
        y2 = self.max_pool(y1)
        y3 = self.down_conv_2(y2) # CCN
        y4 = self.max_pool(y3)
        y5 = self.down_conv_3(y4) # CCN
        y6 = self.max_pool(y5)
        y7 = self.down_conv_4(y6) # CCN
        y8 = self.max_pool(y7)
        y9 = self.down_conv_5(y8) 

        # decoder 
        x = self.up_samp_1(y9)
        x1 = CropIt(y7, x) # crop corresponding encoder part
        x = self.up_conv_1(torch.cat([x,x1],1)) # concatenate the same

        x = self.up_samp_2(x)
        x2 = CropIt(y5, x)
        x = self.up_conv_2(torch.cat([x,x2],1))

        x = self.up_samp_3(x)
        x3 = CropIt(y3, x)
        x = self.up_conv_3(torch.cat([x,x3],1))

        x = self.up_samp_4(y9)
        x4 = CropIt(y1, x)
        x = self.up_conv_4(torch.cat([x,x4],1))

        x = self.out(x)








    
  













































































































































#   def contracting_block(self, in_channels, out_channels, kernel_size=3):
#         """
#         This function creates one contracting block consisting of 2 Convolution Blocks
#         """
#         block = torch.nn.Sequential(
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
#                     torch.nn.ReLU(),
#                     torch.nn.BatchNorm2d(out_channels),
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
#                     torch.nn.ReLU(),
#                     torch.nn.BatchNorm2d(out_channels)
#                 )
#         return block

#     def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This function creates one expansive block consists of 3 Convolution Blocks with 2 being normal and 1 is ConvTranspose
#         """
#         block = torch.nn.Sequential(
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
#                     torch.nn.ReLU(),
#                     torch.nn.BatchNorm2d(mid_channel),
#                     torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
#                     torch.nn.ReLU(),
#                     torch.nn.BatchNorm2d(mid_channel),
#                     torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, 
#                     stride=2, padding=1, output_padding=1)) # upsampling part
#         return  block

#     def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This returns final block consists of 3 Convolution Blocks
#         """
#         block = torch.nn.Sequential(
#                 torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
#                 torch.nn.ReLU(),
#                 torch.nn.BatchNorm2d(mid_channel),
#                 torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
#                 torch.nn.ReLU(),
#                 torch.nn.BatchNorm2d(mid_channel),
#                 torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
#                 torch.nn.ReLU(),
#                 torch.nn.BatchNorm2d(out_channels),
#                 )
#         return  block

#     def __init__(self, in_channel, out_channel):
#         super(UNet, self).__init__()

#         #EnCode Block
#         self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
#         self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode2 = self.contracting_block(64, 128)
#         self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode3 = self.contracting_block(128, 256)
#         self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)


#         # Bottleneck
#         self.bottleneck = torch.nn.Sequential(
#                             torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
#                             torch.nn.ReLU(),
#                             torch.nn.BatchNorm2d(512),
#                             torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
#                             torch.nn.ReLU(),
#                             torch.nn.BatchNorm2d(512),
#                             torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
#                              stride=2, padding=1, output_padding=1))


#         # DeCode Block
#         self.conv_decode3 = self.expansive_block(512, 256, 128)
#         self.conv_decode2 = self.expansive_block(256, 128, 64)
#         self.final_layer = self.final_block(128, 64, out_channel)


#     def crop_and_concat(self, upsampled, bypass, crop=False):
#         """
#         This layer crop the layer from contraction block and concat it with expansive block vector
#         """
#         if crop:
#             c = (bypass.size()[2] - upsampled.size()[2]) // 2
#             bypass = F.pad(bypass, (-c, -c, -c, -c))
#         return torch.cat((upsampled, bypass), 1)


#     def forward(self, x):
#         # Encode
#         encode_block1 = self.conv_encode1(x)
#         encode_pool1 = self.conv_maxpool1(encode_block1)
#         encode_block2 = self.conv_encode2(encode_pool1)
#         encode_pool2 = self.conv_maxpool2(encode_block2)
#         encode_block3 = self.conv_encode3(encode_pool2)
#         encode_pool3 = self.conv_maxpool3(encode_block3)

#         # Bottleneck
#         bottleneck1 = self.bottleneck(encode_pool3)

#         # Decode
#         decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
#         cat_layer2 = self.conv_decode3(decode_block3)
#         decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
#         cat_layer1 = self.conv_decode2(decode_block2)
#         decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
#         final_layer = self.final_layer(decode_block1)
#         return  final_layer





