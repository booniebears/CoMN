import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, layer, kernel_size=3, Dir="Parameters", 
                 conv_connect=False):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.Dir = Dir
        self.layer = layer
        self.conv1 = Conv2d_Q(in_channels, out_channels, (1, kernel_size), layer=layer)
        self.conv2 = Conv2d_Q(in_channels, out_channels, (1, kernel_size), layer=layer+1)
        self.conv3 = Conv2d_Q(in_channels, out_channels, (1, kernel_size), layer=layer+2)
        # self.act = novel_activation(Dir=self.Dir)
        self.act = Sigmoid(inplace=True, layer=1, Dir=self.Dir)
        self.conv_connect = conv_connect
        # self.act = nn.Sigmoid()

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        # print(f"X.shape = {X.shape}") # first call: torch.Size([1, 2, 207, 12])
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        temp = self.conv1(X) + self.act(self.conv2(X))
        # print(f"temp.shape = {temp.shape}") # first call: torch.Size([1, 64, 207, 10])
        if self.layer > 1:
            if self.conv_connect:
                layer_connect('layerconnect', self.layer - 1, self.layer, 'conv1*3', X.size())
            else:
                layer_connect('layerconnect', self.layer - 1, self.layer, 'matmul', X.size())
            layer_connect('layerconnect', self.layer - 1, self.layer + 1, 'conv1*3', X.size())
            layer_connect('layerconnect', self.layer - 1, self.layer + 2, 'conv1*3', X.size())
        # out = F.relu(temp + self.conv3(X))
        relu_func = Relu_S(inplace=True, layer=1, Dir=self.Dir)
        out = relu_func(temp + self.conv3(X))
        layer_connect('layerconnect', self.layer, self.layer + 2, 'conv1*3', temp.size())
        layer_connect('layerconnect', self.layer + 1, self.layer + 2, 'conv1*3', temp.size())
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, layer, Dir="Parameters"):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.Dir = Dir
        self.layer = layer
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels, 
                                   layer=self.layer, Dir=self.Dir, conv_connect=True)
        # self.Theta1.shape = torch.Size([64, 16])
        self.mul1 = MatMul(layer=self.layer+3, Dir=self.Dir)
        self.mul2 = MatMul(layer=self.layer+4, Dir=self.Dir)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels)) # theta(spatial) module
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels, 
                                   layer=self.layer+5, Dir=self.Dir)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        t = t.permute(1, 0, 2, 3)
        t_shape = t.shape
        # print(f"A_hat.shape = {A_hat.shape}") # torch.Size([207, 207])
        # print(f"t.shape = {t.shape}") # torch.Size([207, 1, 10, 64])
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t])
        # TODO: layerconnect not correct.
        layer_connect('layerconnect', self.layer + 2, self.layer + 3, 'conv1*3', t.size())
        lfs = self.mul1(A_hat, t.permute(1, 0, 2, 3).reshape(t.shape[0], -1)).reshape(t_shape).permute(1, 0, 2, 3)
        # print(f"lfs.shape = {lfs.shape}") # torch.Size([1, 207, 10, 64])
        relu_func = Relu_S(inplace=True, layer=1, Dir=self.Dir)
        layer_connect('layerconnect', self.layer + 3, self.layer + 4, 'matmul', lfs.size())
        t2 = relu_func(self.mul2(lfs, self.Theta1))
        # print(f"t2.shape = {t2.shape}") # torch.Size([1, 207, 10, 16])
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output,Dir="Parameters"):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.Dir = Dir
        self.layer = 1
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, 
                                 layer=self.layer, Dir=self.Dir)
        self.layer += 8
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes,
                                 layer=self.layer, Dir=self.Dir)
        self.layer += 8
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64, 
                                       layer=self.layer, Dir=self.Dir, conv_connect=True)
        self.layer += 3
        self.fully = Linear_Q((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output, layer=self.layer, Dir=self.Dir)
        # print(f"num_timesteps_input = {num_timesteps_input}")
        # print(f"num_timesteps_output = {num_timesteps_output}")

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        # print(f"X.shape = {X.shape}") # torch.Size([1, 207, 12, 2])
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        layer_connect('layerconnect', self.layer - 1, self.layer, 'fc', out3.size())
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4
