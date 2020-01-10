import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

##################################################################################################
## Unet model
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True)       )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)  )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class unet(nn.Module):
    def __init__(self):
        super(unet,self).__init__()
        
        self.inc = double_conv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out1 = nn.Conv2d(64, 1, 3, padding=1)
        
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)
        
        # x = self.sigmoid(x)
        x = self.tanh(x)

        return x
##################################################################################################
##################################################################################################
## Unet Batch norm model
class double_conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class down_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_bn, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_bn(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_bn(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_bn, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_bn(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class unet_bn(nn.Module):
    def __init__(self):
        super(unet_bn,self).__init__()
        
        self.inc = double_conv_bn(1, 64)
        self.down1 = down_bn(64, 128)
        self.down2 = down_bn(128, 256)
        self.down3 = down_bn(256, 512)
        self.down4 = down_bn(512, 512)
        
        self.up1 = up_bn(1024, 256)
        self.up2 = up_bn(512, 128)
        self.up3 = up_bn(256, 64)
        self.up4 = up_bn(128, 64)
        self.out1 = nn.Conv2d(64, 1, 3, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)
        
        # x = self.sigmoid(x)
        x = self.tanh(x)

        return x
##################################################################################################
## Unet dropout model
class double_conv_d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Dropout2d(p=0.3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.Dropout2d(p=0.3),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class down_d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_d, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_d(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_d(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_d, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_d(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class unet_d(nn.Module):
    def __init__(self):
        super(unet_d,self).__init__()
        
        self.inc = double_conv_d(1, 64)
        self.down1 = down_d(64, 128)
        self.down2 = down_d(128, 256)
        self.down3 = down_d(256, 512)
        self.down4 = down_d(512, 512)
        
        self.up1 = up_d(1024, 256)
        self.up2 = up_d(512, 128)
        self.up3 = up_d(256, 64)
        self.up4 = up_d(128, 64)
        self.out1 = nn.Conv2d(64, 1, 3, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)
        
        # x = self.sigmoid(x)
        x = self.tanh(x)

        return x
##################################################################################################