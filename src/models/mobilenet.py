"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn
from retrain_modules import *

class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size,layer,Dir="Parameters",**kwargs):
        super().__init__()
        self.Dir = Dir
        self.depthwise = Sequential_S(
            Conv2d_Q(
                input_channels,
                input_channels,
                kernel_size,layer=layer,
                groups=input_channels,
                Dir=self.Dir,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            Relu_S(inplace=True,layer=layer,Dir=self.Dir),layer=layer,Dir=self.Dir
        )
        layer=layer+1
        self.pointwise = Sequential_S(
            Conv2d_Q(input_channels, output_channels, 1,layer=layer,Dir=self.Dir),
            nn.BatchNorm2d(output_channels),
            Relu_S(inplace=True,layer=layer,Dir=self.Dir),layer=layer,Dir=self.Dir
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size,layer,Dir="Parameters",**kwargs):

        super().__init__()
        self.Dir = Dir
        self.conv = Conv2d_Q(
            input_channels, output_channels, kernel_size,layer=layer,Dir=self.Dir,**kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = Relu_S(inplace=True,layer=layer,Dir=self.Dir)
        self.layer = layer

    def forward(self, x):
        layer_connect('layerconnect', self.layer - 1, self.layer, 'conv3*3', x.size(),Dir=self.Dir)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=10,Dir="Parameters"):
       super().__init__()
       layer = 1
       alpha = width_multiplier
       self.Dir = Dir
       self.stem = nn.Sequential(
           BasicConv2d(3, int(32 * alpha), 3, layer=layer,padding=1, bias=False,Dir=self.Dir),

           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,layer=layer+1,
               padding=1,
               bias=False,
               Dir=self.Dir
           )
       )
       layer = layer + 3
       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,layer=layer,
               stride=2,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,layer=layer+2,
               padding=1,
               bias=False,
               Dir=self.Dir
           )
       )
       layer = layer + 4
       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,layer=layer,
               stride=2,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,layer=layer+2,
               padding=1,
               bias=False,
               Dir=self.Dir
           )
       )
       layer = layer + 4
       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,layer=layer,
               stride=2,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,layer=layer+2,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,layer=layer+4,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,layer=layer+6,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,layer=layer+8,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,layer=layer+10,
               padding=1,
               bias=False,
               Dir=self.Dir
           )
       )
       layer = layer + 12
       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,layer=layer,
               stride=2,
               padding=1,
               bias=False,
               Dir=self.Dir
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,layer=layer+2,
               padding=1,
               bias=False,
               Dir=self.Dir
           )
       )
       layer = layer + 4
       self.layer = layer
       self.fc = Linear_Q(int(1024 * alpha), class_num,layer=layer,Dir=self.Dir)
       self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        layer_connect('layerconnect', self.layer - 1, self.layer, 'fc', x.size(),Dir=self.Dir)
        x = self.fc(x)
        return x


def mobilenet(Dir="Parameters"):
    return MobileNet(width_multiplier=1, class_num=10,Dir=Dir)

