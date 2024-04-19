"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from retrain_modules import *
cfg = {
    '11' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M'],
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'C' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, cfg, num_class=1000,Dir="Parameters"):
        super().__init__()
        layer = 1
        self.Dir = Dir
        [self.features,layer] = self.make_layers(cfg, layer,batch_norm=True)

        self.classifier = Sequential_S(

            Linear_Q(512, 512,layer=layer,Dir=self.Dir),
            Relu_S(inplace=True,layer=layer,Dir=self.Dir),
            nn.Dropout(),

            Linear_Q(512, 512,layer=layer+1,Dir=self.Dir),
            Relu_S(inplace=True,layer=layer+1,Dir=self.Dir),
            nn.Dropout(),

            Linear_Q(512, num_class,layer=layer+2,Dir=self.Dir),
            layer = layer,
            Dir=self.Dir
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output
    def make_layers(self,cfg, layer,batch_norm=True):
        layers = []
        inital_layer = layer
        input_channel = 3
        # print(f"In vgg11 make_layers, layer = {layer}") # layer = 1
        for l in cfg:
            if l == 'M':
                layers += [Maxpool_S(kernel_size=2, stride=2, layer=layer-1,Dir=self.Dir)]
                continue

            layers += [Conv2d_Q(input_channel, l, kernel_size=3, padding=1,layer=layer,Dir=self.Dir)]
            layer = layer + 1
            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [Relu_S(inplace=True,layer=layer-1,Dir=self.Dir)]
            input_channel = l

        return Sequential_S(*layers,layer=inital_layer,Dir=self.Dir),layer



def vgg11(Dir="Parameters"):
    return VGG(cfg['A'],num_class=10,Dir=Dir)

def vgg13(Dir="Parameters"):
    return VGG(cfg['B'],num_class=10,Dir=Dir)

def vgg16(Dir="Parameters"):
    return VGG(cfg['C'],num_class=10,Dir=Dir)

def vgg19(Dir="Parameters"):
    return VGG(cfg['D'],num_class=10,Dir=Dir)


