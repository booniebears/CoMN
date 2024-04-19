"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn
from retrain_modules import *

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj,layer):
        super().__init__()

        #1x1conv branch
        self.b1 = Sequential_S(
            Conv2d_Q(input_channels, n1x1, kernel_size=1,layer=layer),
            nn.BatchNorm2d(n1x1),
            Relu_S(inplace=True,layer=layer),layer=layer
        )
        layer = layer + 1
        #1x1conv -> 3x3conv branch
        self.b2 = Sequential_S(
            Conv2d_Q(input_channels, n3x3_reduce, kernel_size=1,layer = layer),
            nn.BatchNorm2d(n3x3_reduce),
            Relu_S(inplace=True,layer=layer),
            Conv2d_Q(n3x3_reduce, n3x3, kernel_size=3, padding=1,layer = layer + 1),
            nn.BatchNorm2d(n3x3),
            Relu_S(inplace=True,layer=layer+1),layer=layer
        )
        layer = layer + 2
        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = Sequential_S(
            Conv2d_Q(input_channels, n5x5_reduce, kernel_size=1,layer = layer),
            nn.BatchNorm2d(n5x5_reduce),
            Relu_S(inplace=True,layer=layer),
            Conv2d_Q(n5x5_reduce, n5x5, kernel_size=3, padding=1,layer = layer+1),
            nn.BatchNorm2d(n5x5, n5x5),
            Relu_S(inplace=True,layer=layer+1),
            Conv2d_Q(n5x5, n5x5, kernel_size=3, padding=1,layer = layer + 2),
            nn.BatchNorm2d(n5x5),
            Relu_S(inplace=True,layer=layer+2),layer=layer
        )
        layer = layer + 3

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = Sequential_S(
            Maxpool_S(3, stride=1, padding=1,layer=layer-1),
            Conv2d_Q(input_channels, pool_proj, kernel_size=1,layer = layer),
            nn.BatchNorm2d(pool_proj),
            Relu_S(inplace=True,layer=layer),layer=layer
        )


    def forward(self, x):
        output = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return output


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        layer = 1
        self.prelayer = Sequential_S(
            Conv2d_Q(3, 192, kernel_size=3, padding=1,layer = layer),
            nn.BatchNorm2d(192),
            Relu_S(inplace=True,layer=layer),layer=layer
        )
        layer = layer + 1
        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32,layer=layer)
        layer = layer + 7
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64,layer=layer)
        layer = layer + 7
        #"""In general, an Inception network is a network consisting of
        #modules of the above type stacked upon each other, with occasional
        #max-pooling layers with stride 2 to halve the resolution of the
        #grid"""
        self.maxpool = Maxpool_S(3, stride=2, padding=1,layer=layer-1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64,layer=layer)
        layer = layer + 7
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64,layer=layer)
        layer = layer + 7
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64,layer=layer)
        layer = layer + 7
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64,layer=layer)
        layer = layer + 7
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128,layer=layer)
        layer = layer + 7
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128,layer=layer)
        layer = layer + 7
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128,layer=layer)
        layer = layer + 7

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = Linear_Q(1024, num_class,layer=layer)

    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)

        output = self.maxpool(output)

        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)

        output = self.maxpool(output)

        output = self.a5(output)
        output = self.b5(output)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        output = self.avgpool(output)
        output = self.dropout(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)

        return output

def googlenet(num_class):
    return GoogleNet(num_class=num_class)


