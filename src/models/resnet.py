"""resnet in pytorch

#
#
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
#
#     Deep Residual Learning for Image Recognition
#     https://arxiv.org/abs/1512.03385v1
# """
#
import torch
import torch.nn as nn
from retrain_modules import *
from Parameters import *

#
def conv3x3(in_planes, out_planes, layer,stride=1, groups=1, dilation=1,Dir="Parameters"):
    """3x3 convolution with padding"""
    return Conv2d_Q(in_planes, out_planes, kernel_size=3, stride=stride,layer=layer,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,Dir=Dir)


def conv1x1(in_planes, out_planes, layer,stride=1,Dir="Parameters"):
    """1x1 convolution"""
    return Conv2d_Q(in_planes, out_planes,layer=layer, kernel_size=1, stride=stride, bias=False,
                    Dir=Dir)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, layer, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,Dir="Parameters"):
        super(BasicBlock, self).__init__()
        self.layer = layer
        self.Dir = Dir
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, layer,stride,Dir=self.Dir)
        layer = layer + 1
        self.bn1 = norm_layer(planes)
        self.relu = Relu_S(inplace=True, layer=layer,Dir=self.Dir)
        self.conv2 = conv3x3(planes, planes,layer=layer,Dir=self.Dir)
        layer = layer + 1
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            self.layer = self.layer - 1
            thislayer = 1
        else:
            thislayer = 0

        layer_connect('layerconnect', self.layer - 1, self.layer + thislayer, 'conv3*3', x.size(),Dir=self.Dir)
        if self.downsample is not None:
            identity = self.downsample(x)
            layer_connect('layerconnect', self.layer-1 , self.layer, 'residual_conv1*1', x.size(),Dir=self.Dir)
            layer_connect('layerconnect', self.layer, self.layer + thislayer + 1, 'residual_conv1*1', x.size(),Dir=self.Dir)
        else:
            layer_connect('layerconnect', self.layer - 1, self.layer + thislayer + 1, 'shortcut', x.size(),Dir=self.Dir)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        layer_connect('layerconnect', self.layer + thislayer, self.layer + thislayer + 1, 'conv3*3', out.size(),Dir=self.Dir)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,Dir="Parameters"):
        super(Bottleneck, self).__init__()
        self.layer = layer
        self.Dir = Dir
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width,layer=layer,Dir=self.Dir)
        layer = layer + 1
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, layer, stride, groups, dilation,Dir=self.Dir)
        layer = layer + 1
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion,layer=layer,Dir=self.Dir)
        layer = layer + 1
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = Relu_S(inplace=True, layer=layer,Dir=self.Dir)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            self.layer = self.layer - 1
            thislayer = 1
            identity = self.downsample(x)
            layer_connect('layerconnect', self.layer - 1, self.layer, 'conv3*3', x.size(),Dir=self.Dir)
            layer_connect('layerconnect', self.layer, self.layer + thislayer + 2, 'residual_conv1*1', x.size(),Dir=self.Dir)
        else:
            thislayer = 0
            layer_connect('layerconnect', self.layer - 1, self.layer + thislayer + 2, 'shortcut', x.size(),Dir=self.Dir)

        layer_connect('layerconnect',self.layer - 1, self.layer + thislayer, 'conv1*1', x.size(),Dir=self.Dir)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        layer_connect('layerconnect',self.layer + thislayer, self.layer + thislayer + 1, 'conv3*3', out.size(),Dir=self.Dir)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        layer_connect('layerconnect',self.layer + thislayer + 1, self.layer + thislayer + 2, 'conv1*1', out.size(),Dir=self.Dir)
        out = self.conv3(out)
        out = self.bn3(out)


        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,Dir="Parameters"):
        super(ResNet, self).__init__()

        layer = 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.Dir = Dir

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d_Q(3, self.inplanes, kernel_size=3, stride=1, padding=1,layer=layer,
                               bias=False,Dir=self.Dir)
        layer = layer + 1
        self.bn1 = norm_layer(self.inplanes)
        self.relu = Relu_S(True, layer=layer,Dir=self.Dir)
        self.maxpool = Maxpool_S(kernel_size=2, stride=2, padding=1,layer=layer,Dir=self.Dir)
        [self.layer1,layer] = self._make_layer(block, 64, layers[0],layer=layer)
        [self.layer2,layer] = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],layer=layer)
        [self.layer3,layer] = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],layer=layer)
        [self.layer4,layer] = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],layer=layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear_Q(512 * block.expansion, num_classes,layer=layer,Dir=self.Dir)
        self.layer = layer
        for m in self.modules():
            if isinstance(m, Conv2d_Q):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, layer,stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, layer,stride,Dir=self.Dir),
                norm_layer(planes * block.expansion),
            )
            layer = layer + 1




        layers = []
        insert_layer = block(self.inplanes, planes,layer,stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,Dir=self.Dir)
        layers.append(insert_layer)
        if isinstance(insert_layer,BasicBlock):
            layer = layer + 2

        if isinstance(insert_layer,Bottleneck):
            layer = layer + 3
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            insert_layer = block(self.inplanes, planes, groups=self.groups, layer=layer,
                  base_width=self.base_width, dilation=self.dilation,
                  norm_layer=norm_layer,Dir=self.Dir)
            layers.append(insert_layer)
            if isinstance(insert_layer, BasicBlock):
                layer = layer + 2
            if isinstance(insert_layer, Bottleneck):
                layer = layer + 3

        return nn.Sequential(*layers),layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        layer_connect('layerconnect',self.layer-1,self.layer,'fc',x.size(),Dir=self.Dir)
        x = self.fc(x)

        return x

# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34
#
#     """
#
#     #BasicBlock and BottleNeck block
#     #have different output size
#     #we use class attribute expansion
#     #to distinct
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         #residual function
#         self.residual_function = nn.Sequential(
#             # Myexp(),
#             Conv2d_Q(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU6(inplace=True),
#             # Myexp(),
#             Conv2d_Q(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )
#
#         #shortcut
#         self.shortcut = nn.Sequential(
#             # nn.BatchNorm2d(out_channels)
#             # Myexp(),
#         )
#
#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 # Myexp(),
#                 Conv2d_Q(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )
#
#     def forward(self, x):
#         return nn.ReLU6(inplace=True)(self.residual_function(x) + self.shortcut(x))
#
#         # output = (self.residual_function(x) + self.shortcut(x))
#         # return output
#
# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers
#
#     """
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             # Myexp(),
#             Conv2d_Q(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU6(inplace=True),
#             # Myexp(),
#             Conv2d_Q(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU6(inplace=True),
#             # Myexp(),
#             Conv2d_Q(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )
#
#         self.shortcut = nn.Sequential(
#
#         )
#
#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 # Myexp(),
#                 Conv2d_Q(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1,  bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )
#
#     def forward(self, x):
#         return nn.ReLU6(inplace=True)(self.residual_function(x) + self.shortcut(x))
#         # return (self.residual_function(x) + self.shortcut(x))
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, num_block, num_classes=1000):
#         super().__init__()
#
#         self.in_channels = 64
#
#         self.conv1 = nn.Sequential(
#             # Myexp(),
#             Conv2d_Q(3, 64, kernel_size=3,  padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU6(inplace=True)
#         )
#         #we use a different inputsize than the original paper
#         #so conv2_x's stride is 1
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = Linear_Q(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block
#
#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer
#
#         Return:
#             return a resnet layer
#         """
#
#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         output = self.conv3_x(output)
#         output = self.conv4_x(output)
#         output = self.conv5_x(output)
#         output = self.avg_pool(output)
#         output = output.view(output.size(0), -1)
#         # output = torch.exp(0.3 * output) - 1
#         output = self.fc(output)
#
#         return output
#
def resnet18(Dir="Parameters"):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=10,Dir=Dir)

def resnet34(Dir="Parameters"):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=10,Dir=Dir)

def resnet50(Dir="Parameters"):
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=10,Dir=Dir)

def resnet101(Dir="Parameters"):
    """ return a ResNet 101 object
    """
    return ResNet(Bottleneck, [3, 4, 23, 3],num_classes=10,Dir=Dir)

def resnet152(Dir="Parameters"):
    """ return a ResNet 152 object
    """
    return ResNet(Bottleneck, [3, 8, 36, 3],num_classes=10,Dir=Dir)
# import torch
# import torch.nn as nn
# from .utils import load_state_dict_from_url

#
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']
#
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }
#
#
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return Conv2d_Q(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return Conv2d_Q(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = Conv2d_Q(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = Linear_Q(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, Conv2d_Q):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#
# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     # if pretrained:
#     #     state_dict = load_state_dict_from_url(model_urls[arch],
#     #                                           progress=progress)
#     #     model.load_state_dict(state_dict)
#     return model
#
#
# def resnet18(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)
#
#
# def resnet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet50(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet101(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet152(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)
#
#
# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)



