import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


def double_conv(in_c, out_c, mid_channels=None):
    if not mid_channels:
        mid_channels = out_c
    conv = nn.Sequential(
        nn.Conv2d(in_c, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
    return conv


def resize_up_conv_image(down_conv_x1, up_conv_x2):
    diffY = up_conv_x2.size()[2] - down_conv_x1.size()[2]
    diffX = up_conv_x2.size()[3] - down_conv_x1.size()[3]

    down_conv_x1 = F.pad(down_conv_x1, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
    return torch.cat([up_conv_x2, down_conv_x1], dim=1)


class Down(nn.Module):
    """Downscaling the size and doubling channels"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling the size and down scalling channels"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up_trans = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_conv = double_conv(in_channels, out_channels, in_channels // 2)
        else:
            self.up_trans = nn.ConvTranspose2d(in_channels=in_channels,
                                               out_channels=in_channels // 2,
                                               kernel_size=2,
                                               stride=2)
            self.up_conv = double_conv(in_channels, out_channels)

    def forward(self, down_conv_x1, up_conv_x2):
        # x = self.up_trans(down_conv_x1)
        # y = crop_img(down_conv_x1, up_conv_x2)  #croped x7
        # x = self.up_conv(torch.cat([x,y], 1))
        # return x

        down_conv_x1 = self.up_trans(down_conv_x1)
        x = resize_up_conv_image(down_conv_x1, up_conv_x2)
        return self.up_conv(x)


class Unet(nn.Module):
    def __init__(self, num_channels=6, num_classes=1, bilinear=True):
        super(Unet, self).__init__()

        factor = 2 if bilinear else 1

        self.num_classes = num_classes
        self.num_channels = num_channels

        # encoder
        self.down_conv_1 = double_conv(self.num_channels, 32)

        # ***********Depth convoluton part*****************
        self.down2_d = Down(32, 64)
        self.down3_d = Down(64, 128)
        self.down4_d = Down(128, 256)
        self.down5_d = Down(256, 512 // factor)

        self.up1_d = Up(512, 256 // factor, bilinear)
        self.up2_d = Up(256, 128 // factor, bilinear)
        self.up3_d = Up(128, 64 // factor, bilinear)
        self.up4_d = Up(64, 32, bilinear)

        self.out_d = nn.Conv2d(
            in_channels=32,
            out_channels=self.num_classes,
            kernel_size=1
        )

        # ***********Mask convoluton part*****************
        self.down2_m = Down(32, 64)
        self.down3_m = Down(64, 128)
        self.down4_m = Down(128, 256)
        self.down5_m = Down(256, 512 // factor)

        self.up1_m = Up(512, 256 // factor, bilinear)
        self.up2_m = Up(256, 128 // factor, bilinear)
        self.up3_m = Up(128, 64 // factor, bilinear)
        self.up4_m = Up(64, 32, bilinear)

        self.out_m = nn.Conv2d(
            in_channels=32,
            out_channels=self.num_classes,
            kernel_size=1
        )

    def forward(self, image):
        # bs, c, h, w
        # encoding starts
        image = image
        x1 = self.down_conv_1(image)  #

        # *************depth starts*******
        # encoder
        x2 = self.down2_d(x1)
        x3 = self.down3_d(x2)
        x4 = self.down4_d(x3)
        x5 = self.down5_d(x4)

        # decoder
        x_d = self.up1_d(x5, x4)
        x_d = self.up2_d(x_d, x3)
        x_d = self.up3_d(x_d, x2)
        x_d = self.up4_d(x_d, x1)
        dense_output = self.out_d(x_d)

        # *************mask starts*******
        # encoder
        x2 = self.down2_m(x1)
        x3 = self.down3_m(x2)
        x4 = self.down4_m(x3)
        x5 = self.down5_m(x4)

        # decoder
        x_m = self.up1_m(x5, x4)
        x_m = self.up2_m(x_m, x3)
        x_m = self.up3_m(x_m, x2)
        x_m = self.up4_m(x_m, x1)
        mask_output = self.out_m(x_m)
        return dense_output, mask_output