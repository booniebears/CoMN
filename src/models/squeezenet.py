"""squeezenet in pytorch



[1] Song Han, Jeff Pool, John Tran, William J. Dally

    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn
from retrain_modules import *

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel,layer):

        super().__init__()
        self.layer = layer
        self.squeeze = nn.Sequential(
            Conv2d_Q(in_channel, squzee_channel, 1,layer=layer),
            nn.BatchNorm2d(squzee_channel),
            Relu_S(inplace=True,layer=layer),
        )

        self.expand_1x1 = nn.Sequential(
            Conv2d_Q(squzee_channel, int(out_channel / 2), 1,layer=layer+1),
            nn.BatchNorm2d(int(out_channel / 2)),
            Relu_S(inplace=True,layer=layer+1)
        )

        self.expand_3x3 = nn.Sequential(
            Conv2d_Q(squzee_channel, int(out_channel / 2), 3, padding=1,layer=layer+2),
            nn.BatchNorm2d(int(out_channel / 2)),
            Relu_S(inplace=True,layer=layer+2)
        )

    def forward(self, x):
        layer_connect('layerconnect', self.layer - 1, self.layer, 'conv', x.size())
        x = self.squeeze(x)
        layer_connect('layerconnect', self.layer , self.layer+1, 'conv', x.size())
        layer_connect('layerconnect', self.layer, self.layer + 2, 'conv', x.size())
        out1 = self.expand_1x1(x)
        out2 = self.expand_3x3(x)
        x = torch.cat([out1,out2], 1)
        layer_connect('layerconnect', self.layer + 1, self.layer + 2, 'add', out1.size())

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=100):

        super().__init__()
        layer = 1
        self.layer = layer
        self.stem = nn.Sequential(
            Conv2d_Q(3, 96, 3, padding=1,layer=layer),
            nn.BatchNorm2d(96),
            Relu_S(inplace=True,layer=layer),
            Maxpool_S(2, 2,layer=layer)
        )
        layer = layer + 1
        self.fire2 = Fire(96, 128, 16,layer=layer)
        layer = layer + 3
        self.fire3 = Fire(128, 128, 16,layer=layer)
        layer = layer + 3
        self.fire4 = Fire(128, 256, 32,layer=layer)
        layer = layer + 3
        self.maxpool1 = Maxpool_S(2, 2, layer=layer)
        self.fire5 = Fire(256, 256, 32,layer=layer)
        layer = layer + 3
        self.fire6 = Fire(256, 384, 48,layer=layer)
        layer = layer + 3
        self.fire7 = Fire(384, 384, 48,layer=layer)
        layer = layer + 3
        self.fire8 = Fire(384, 512, 64,layer=layer)
        layer = layer + 3
        self.maxpool2 = Maxpool_S(2, 2, layer=layer)
        self.fire9 = Fire(512, 512, 64,layer=layer)
        layer = layer + 3
        self.conv10 = Conv2d_Q(512, class_num, 1,layer=layer)
        self.avg = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        layer_connect('layerconnect', self.layer + 3, self.layer + 6, 'shortcut', f2.size())
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool1(f4)
        layer_connect('layerconnect', self.layer + 9, self.layer + 12, 'shortcut', f4.size())
        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        layer_connect('layerconnect', self.layer + 15, self.layer + 18, 'shortcut', f6.size())
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool2(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return x

def squeezenet(num_class):
    return SqueezeNet(class_num=num_class)
