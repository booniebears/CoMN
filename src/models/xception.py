"""xception in pytorch


[1] François Chollet

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357
"""

import torch
import torch.nn as nn
from retrain_modules import *

class SeperableConv2d(nn.Module):

    #***Figure 4. An “extreme” version of our Inception module,
    #with one spatial convolution per output channel of the 1x1
    #convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, layer,**kwargs):

        super().__init__()
        self.depthwise = Conv2d_Q(
            input_channels,
            input_channels,
            kernel_size,layer=layer,
            groups=input_channels,
            bias=False,
            **kwargs
        )

        self.pointwise = Conv2d_Q(input_channels, output_channels, 1, bias=False,layer=layer+1)
        self.layer = layer

    def forward(self, x):

        x = self.depthwise(x)
        layer_connect('layerconnect', self.layer , self.layer + 1, 'conv', x.size())
        x = self.pointwise(x)

        return x

class EntryFlow(nn.Module):

    def __init__(self,layer):

        super().__init__()
        self.layer = layer
        self.conv1 = Sequential_S(
            Conv2d_Q(3, 32, 3, padding=1, bias=False,layer=layer),
            nn.BatchNorm2d(32),
            Relu_S(inplace=True,layer=layer),layer=layer
        )

        self.conv2 = nn.Sequential(
            Conv2d_Q(32, 64, 3, padding=1, bias=False,layer=layer+1),
            nn.BatchNorm2d(64),
            Relu_S(inplace=True,layer=layer+1)
        )

        self.conv3_0_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1,layer=layer+2),
            nn.BatchNorm2d(128),
            Relu_S(inplace=True,layer=layer+3),
        )
        self.conv3_1_residual = nn.Sequential(
            SeperableConv2d(128, 128, 3, padding=1,layer=layer+4),
            nn.BatchNorm2d(128),
            Maxpool_S(3, stride=2, padding=1,layer=layer+5)
        )

        self.conv3_shortcut = nn.Sequential(
            Conv2d_Q(64, 128, 1, stride=2,layer=layer+6),
            nn.BatchNorm2d(128),
        )

        self.conv4_0_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1,layer=layer+7),
            nn.BatchNorm2d(256),
            Relu_S(inplace=True,layer=layer+8),
        )
        self.conv4_1_residual = nn.Sequential(
            SeperableConv2d(256, 256, 3, padding=1,layer=layer+9),
            nn.BatchNorm2d(256),
            Maxpool_S(3, stride=2, padding=1,layer=layer+10)
        )

        self.conv4_shortcut = nn.Sequential(
            Conv2d_Q(128, 256, 1, stride=2,layer=layer+11),
            nn.BatchNorm2d(256),
        )

        #no downsampling
        self.conv5_0_residual = nn.Sequential(
            Relu_S(inplace=True,layer=layer+11),
            SeperableConv2d(256, 728, 3, padding=1,layer=layer+12),
            nn.BatchNorm2d(728),
            Relu_S(inplace=True,layer=layer+13),
        )
        self.conv5_1_residual = nn.Sequential(
            SeperableConv2d(728, 728, 3, padding=1,layer=layer+14),
            nn.BatchNorm2d(728),
            Maxpool_S(3, 1, padding=1,layer=layer+15)
        )

        #no downsampling
        self.conv5_shortcut = nn.Sequential(
            Conv2d_Q(256, 728, 1,layer=layer+16),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        layer_connect('layerconnect', self.layer, self.layer + 1, 'conv', x.size())
        x = self.conv2(x)
        layer_connect('layerconnect', self.layer + 1, self.layer + 2, 'conv', x.size())
        residual_1 = self.conv3_0_residual(x)
        layer_connect('layerconnect', self.layer+3, self.layer + 4, 'conv', residual_1.size())
        residual = self.conv3_1_residual(residual_1)
        layer_connect('layerconnect', self.layer + 1, self.layer + 6, 'conv', x.size())
        shortcut = self.conv3_shortcut(x)
        layer_connect('layerconnect', self.layer + 5, self.layer + 6, 'conv', residual.size())
        x = residual + shortcut
        layer_connect('layerconnect', self.layer + 6, self.layer + 7, 'conv', x.size())
        residual_1 = self.conv4_0_residual(x)
        layer_connect('layerconnect', self.layer + 8, self.layer + 9, 'conv', residual_1.size())
        residual = self.conv4_1_residual(residual_1)
        layer_connect('layerconnect', self.layer + 6, self.layer + 11, 'conv', x.size())
        shortcut = self.conv4_shortcut(x)
        layer_connect('layerconnect', self.layer + 10, self.layer + 11, 'conv', residual.size())
        x = residual + shortcut
        layer_connect('layerconnect', self.layer + 11, self.layer + 12, 'conv', x.size())
        residual_1 = self.conv5_0_residual(x)
        layer_connect('layerconnect', self.layer + 13, self.layer + 14, 'conv', residual_1.size())
        residual = self.conv5_1_residual(residual_1)
        layer_connect('layerconnect', self.layer + 11, self.layer + 16, 'conv', x.size())
        shortcut = self.conv5_shortcut(x)
        layer_connect('layerconnect', self.layer + 15, self.layer + 16, 'conv', residual.size())
        x = residual + shortcut

        return x

class MiddleFLowBlock(nn.Module):

    def __init__(self,layer):
        super().__init__()
        self.layer = layer

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            Relu_S(inplace=True,layer=layer-1),
            SeperableConv2d(728, 728, 3, padding=1,layer=layer),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            Relu_S(inplace=True,layer=layer+1),
            SeperableConv2d(728, 728, 3, padding=1,layer=layer+2),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            Relu_S(inplace=True,layer=layer+3),
            SeperableConv2d(728, 728, 3, padding=1,layer=layer+4),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        layer_connect('layerconnect', self.layer -1 , self.layer , 'conv', x.size())
        residual = self.conv1(x)
        layer_connect('layerconnect', self.layer + 1, self.layer + 2, 'conv', residual.size())
        residual = self.conv2(residual)
        layer_connect('layerconnect', self.layer + 3, self.layer + 4, 'conv', residual.size())
        residual = self.conv3(residual)
        layer_connect('layerconnect', self.layer - 1, self.layer + 5, 'conv', x.size())
        shortcut = self.shortcut(x)

        return shortcut + residual

class MiddleFlow(nn.Module):
    def __init__(self, block,layer):
        super().__init__()

        #"""then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8,layer)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times,layer):
        flows = []
        for i in range(times):
            flows.append(block(layer))
            layer = layer + 6

        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self,layer):
        super().__init__()
        self.layer = layer
        self.residual_1 = nn.Sequential(
            Relu_S(inplace=True,layer=layer-1),
            SeperableConv2d(728, 728, 3, padding=1,layer=layer),
            nn.BatchNorm2d(728),
            Relu_S(inplace=True,layer=layer+1),
        )
        self.residual_2 = nn.Sequential(
            SeperableConv2d(728, 1024, 3, padding=1,layer=layer+2),
            nn.BatchNorm2d(1024),
            Maxpool_S(3, stride=2, padding=1,layer=layer+3)
        )

        self.shortcut = nn.Sequential(
            Conv2d_Q(728, 1024, 1, stride=2,layer=layer+4),
            nn.BatchNorm2d(1024)
        )

        self.conv_1 = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1,layer=layer+5),
            nn.BatchNorm2d(1536),
            Relu_S(inplace=True,layer=layer+6),
        )
        self.conv_2 = nn.Sequential(
            SeperableConv2d(1536, 2048, 3, padding=1,layer=layer+7),
            nn.BatchNorm2d(2048),
            Relu_S(inplace=True,layer=layer+8)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        layer_connect('layerconnect', self.layer - 1, self.layer, 'convt', x.size())
        residual = self.residual_1(x)
        layer_connect('layerconnect', self.layer + 1, self.layer + 2, 'conv', residual.size())
        residual = self.residual_2(residual)
        layer_connect('layerconnect', self.layer - 1, self.layer + 4, 'shortcut', x.size())
        shortcut = self.shortcut(x)
        layer_connect('layerconnect', self.layer + 3, self.layer + 4, 'shortcut', residual.size())
        output = shortcut + residual
        layer_connect('layerconnect', self.layer + 4, self.layer + 5, 'conv', output.size())
        output = self.conv_1(output)
        layer_connect('layerconnect', self.layer + 6, self.layer + 7, 'conv', output.size())
        output = self.conv_2(output)
        output = self.avgpool(output)

        return output

class Xception(nn.Module):

    def __init__(self, block, num_class=100):
        super().__init__()
        layer = 1

        self.entry_flow = EntryFlow(layer)
        layer = layer+17
        self.middel_flow = MiddleFlow(block,layer)
        layer = layer + 6*8
        self.exit_flow = ExitFLow(layer)
        layer = layer+9
        self.layer = layer

        self.fc = Linear_Q(2048, num_class,layer=layer)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        layer_connect('layerconnect', self.layer-1, self.layer, 'conv', x.size())
        x = self.fc(x)

        return x

def xception(num_class):
    return Xception(MiddleFLowBlock,num_class=num_class)


