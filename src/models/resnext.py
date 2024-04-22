"""resnext in pytorch



[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.

    Aggregated Residual Transformations for Deep Neural Networks
    https://arxiv.org/abs/1611.05431
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from retrain_modules import *

#only implements ResNext bottleneck c


#"""This strategy exposes a new dimension, which we call “cardinality”
#(the size of the set of transformations), as an essential factor
#in addition to the dimensions of depth and width."""
CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64

#"""The grouped convolutional layer in Fig. 3(c) performs 32 groups
#of convolutions whose input and output channels are 4-dimensional.
#The grouped convolutional layer concatenates them as the outputs
#of the layer."""

class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride,layer):
        super().__init__()
        self.layer = layer

        C = CARDINALITY #How many groups a feature map was splitted into

        #"""We note that the input/output width of the template is fixed as
        #256-d (Fig. 3), We note that the input/output width of the template
        #is fixed as 256-d (Fig. 3), and all widths are dou- bled each time
        #when the feature map is subsampled (see Table 1)."""
        D = int(DEPTH * out_channels / BASEWIDTH) #number of channels per group
        self.split_transforms = Sequential_S(
            Conv2d_Q(in_channels, C * D, kernel_size=1, groups=C, bias=False,layer=layer),
            nn.BatchNorm2d(C * D),
            Relu_S(inplace=True,layer=layer),
            Conv2d_Q(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False,layer=layer+1),
            nn.BatchNorm2d(C * D),
            Relu_S(inplace=True,layer=layer),
            Conv2d_Q(C * D, out_channels * 4, kernel_size=1, bias=False,layer=layer+2),
            nn.BatchNorm2d(out_channels * 4),layer=layer
        )
        layer = layer+3
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                Conv2d_Q(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False,layer=layer),
                nn.BatchNorm2d(out_channels * 4)
            )
        self.relu = Relu_S(inplace=True,layer=layer-1)


    def forward(self, x):
        output = self.split_transforms(x)
        if isinstance(self.shortcut, torch.nn.modules.container.Sequential):
            layer_connect('layerconnect', self.layer - 1, self.layer+3, 'conv1*1', x.size())
            layer_connect('layerconnect', self.layer + 3, self.layer + 2, 'conv1*1', x.size())
        else:
            layer_connect('layerconnect', self.layer - 1, self.layer + 2, 'shortcut', x.size())
        identity = self.shortcut(x)
        output = self.relu(output + identity)
        return output

class ResNext(nn.Module):

    def __init__(self, block, num_blocks, class_names=100):
        super().__init__()
        layer = 1
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            Conv2d_Q(3, 64, 3, stride=1, padding=1, bias=False,layer=layer),
            nn.BatchNorm2d(64),
            Relu_S(inplace=True,layer=layer)
        )
        layer = layer+1

        [self.conv2,layer] = self._make_layer(block, num_blocks[0], 64, 1,layer)
        [self.conv3,layer] = self._make_layer(block, num_blocks[1], 128, 2,layer)
        [self.conv4,layer] = self._make_layer(block, num_blocks[2], 256, 2,layer)
        [self.conv5,layer] = self._make_layer(block, num_blocks[3], 512, 2,layer)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear_Q(512 * 4, 100,layer=layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride,layer):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride

        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            if stride != 1 or self.in_channels != out_channels * 4:
                layers.append(block(self.in_channels, out_channels, stride,layer))
                layer = layer + 4
            else:
                layers.append(block(self.in_channels, out_channels, stride, layer))
                layer = layer + 3
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers),layer

def resnext50(num_class):
    """ return a resnext50(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 6, 3], class_names=num_class)

def resnext101(num_class):
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 23, 3], class_names=num_class)

def resnext152(num_class):
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 36, 3], class_names=num_class)



