import torch
import torch.nn as nn
import torch.nn.functional as func
from retrain_modules import *

###########################################################################
# Activation Function for our novel device. Composed of Sigmoid functions.#
###########################################################################
class novel_activation(nn.Module):
    def __init__(self,layer=1,Dir="Parameters"):
        super().__init__()
        self.layer = layer
        self.Dir = Dir

    def forward(self, input):
    #   print(f"layer = {self.layer}, Input range to novel_activation:", input.min().item(), input.max().item())
        on_off = 20
        level0 = 1.0 / (on_off * on_off)
        level1 = (on_off - 1) / (on_off * on_off)
        level2 = (on_off - 1) / on_off
        gain1 = 8
        gain2 = 8
        left_pos = 0
        right_pos = 2
        sigmoid_func = Sigmoid_S(inplace=True, layer=1, Dir=self.Dir)
        # return level2 * torch.sigmoid(gain1 * (input - left_pos)) + \
        #         level1 * torch.sigmoid(gain2 * (input - right_pos)) + level0
        return level2 * sigmoid_func(gain1 * (input - left_pos)) + \
                level1 * sigmoid_func(gain2 * (input - right_pos)) + level0

class Sigmoid(nn.Sigmoid):
    def __init__(self,inplace, layer,Dir="Parameters"):
        super(Sigmoid, self).__init__()
        self.inplace = inplace
        self.layer = layer
        self.Dir = Dir

    def forward(self, input):
        output = torch.sigmoid(input)
        return output

class Lenet5(nn.Module):
    def __init__(self, Dir = "Parameters"):
        super(Lenet5, self).__init__()
        self.Dir = Dir
        # self.conv1 = Conv2d_Q(3, 6, 5, bias=False, layer=1, Dir=self.Dir) # Lenet5 + CIFAR10
        self.conv1 = Conv2d_Q(1, 6, 5, bias=False, layer=1, Dir=self.Dir) # Lenet5 + MNIST
        self.conv2 = Conv2d_Q(6, 16, 5, bias=False, layer=2, Dir=self.Dir)
        self.fc1 = Linear_Q(16 * 5 * 5, 120,layer=3, Dir=self.Dir)
        self.fc2 = Linear_Q(120, 84, layer=4, Dir=self.Dir)
        self.fc3 = Linear_Q(84, 10, layer=5, Dir=self.Dir)
        self.act = novel_activation(Dir=self.Dir)
        self.act1 = Sigmoid_S(inplace=True, layer=1, Dir=self.Dir)
        self.act2 = Sigmoid_S(inplace=True, layer=2, Dir=self.Dir)
        self.act3 = Sigmoid_S(inplace=True, layer=3, Dir=self.Dir)
        self.act4 = Sigmoid_S(inplace=True, layer=4, Dir=self.Dir)
        self.maxpool1 = Maxpool_S(kernel_size=2, stride=2, layer=1, Dir=self.Dir)
        self.maxpool2 = Maxpool_S(kernel_size=2, stride=2, layer=2, Dir=self.Dir)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        layer_connect('layerconnect', 1, 2, 'conv', x.size(), Dir=self.Dir)
        x = self.maxpool1(x)
        x = self.act2(self.conv2(x))
        layer_connect('layerconnect', 2, 3, 'fc', x.size(), Dir=self.Dir)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.act3(self.fc1(x))
        layer_connect('layerconnect', 3, 4, 'fc', x.size(), Dir=self.Dir)
        x = self.act4(self.fc2(x))
        layer_connect('layerconnect', 4, 5, 'fc', x.size(), Dir=self.Dir)
        x = self.fc3(x)
        return x
