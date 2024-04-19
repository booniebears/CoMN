import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from Parameters import *
import math
import random
from HardwareVMM import *

filename = ''
max_weight = 0.5
max_output = 4
max_input = 3

class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# ********************* Activation quantization ***********************
class activation_quantize(nn.Module):
    def __init__(self, a_bits):
        super().__init__()
        self.a_bits = a_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.a_bits == 32:
            output = input

        elif self.a_bits == 1:
            print('! Binary quantization is not supported !')
            assert self.a_bits != 1
        else:
            scale = float(2 ** self.a_bits - 1) / 2
            # print("adc resolution: ", self.a_bits)
            # if first == 1:
            #     output = output*scale
            # else:
            input = torch.clamp(input,-1*max_input,max_input)
            output = input / max_input * scale
            output = self.round(output)
            output = output * max_input / scale

        return output


# ********************* Weight quantization ***********************
class weight_quantize(nn.Module):
    def __init__(self, w_bits):
        super().__init__()
        self.w_bits = w_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print('! Binary quantization is not supported !')
            assert self.w_bits != 1
        else:
            # output = torch.tanh(input)
            # output = input
            input = torch.clamp(input,-1*max_weight,max_weight) # 归一化-[0,1]


            scale = float(2 ** self.w_bits - 1) / 2
            output = input / max_weight * scale # [-0.5,0.5] -> [-scale,scale]
            output = self.round(output)
            output = output * max_weight / scale # scale back here

            # output = 2 * output - 1
        return output


class Defect(Function):

    @staticmethod
    def forward(self, weight):
        kernel_vector = weight.reshape(weight.shape[0], -1)
        Trainfactors = trainParam(Dir="Parameters")
        
        # The defect of weight of each layer in VGG-11 has been tested.
        # The weights with defect are located in /defect, showing the position 
        # of stuck at '0's and stuck at '1's.
        if Trainfactors["vgg11"]:
            if os.path.isfile(filename):
                # print("=> loading array defect from'{}'".format(filename))
                checkpoint = torch.load(filename)
                defect_pos_0 = checkpoint['defect_pos_0'].cuda()
                defect_pos_1 = checkpoint['defect_pos_1'].cuda()
                defect_neg_0 = checkpoint['defect_neg_0'].cuda()
                defect_neg_1 = checkpoint['defect_neg_1'].cuda()
                kernel_zeros = torch.zeros(kernel_vector.shape).cuda()
                kernel_pos = torch.max(kernel_vector, kernel_zeros)
                kernel_neg = -1 * torch.min(kernel_vector, kernel_zeros)
                kernel_pos = torch.min(defect_pos_0, kernel_pos)
                kernel_pos = torch.max(defect_pos_1, kernel_pos)
                kernel_neg = torch.min(defect_neg_0, kernel_neg)
                kernel_neg = torch.max(defect_neg_1, kernel_neg)
                kernel_vector = kernel_pos - kernel_neg
                weight = kernel_vector.reshape(weight.shape)
            else:
                print("=> no array defect found at '{}'".format(filename))
        else:
            rows = kernel_vector.shape[0]
            cols = kernel_vector.shape[1]
            matrix_ZERO = torch.zeros_like(kernel_vector).cuda()
            matrix_ONE = torch.ones_like(kernel_vector).cuda()
            num_zeros = int(rows * cols * 0.01)
            num_ones = int(rows * cols * 0.01)
            # Four random seeds, for defect_pos_0,defect_pos_1,defect_neg_0,defect_neg_1
            torch.manual_seed(42)
            zero_indices = torch.randperm(rows * cols)[:num_zeros]
            defect_pos_0 = matrix_ONE
            defect_pos_0.view(-1)[zero_indices] = 0
            torch.manual_seed(43)
            one_indices = torch.randperm(rows * cols)[:num_ones]
            defect_pos_1 = matrix_ZERO
            defect_pos_1.view(-1)[one_indices] = 1
            torch.manual_seed(44)
            zero_indices = torch.randperm(rows * cols)[:num_zeros]
            defect_neg_0 = matrix_ONE
            defect_neg_0.view(-1)[zero_indices] = 0
            torch.manual_seed(45)
            one_indices = torch.randperm(rows * cols)[:num_ones]
            defect_neg_1 = matrix_ZERO
            defect_neg_1.view(-1)[one_indices] = 1
            
            kernel_zeros = torch.zeros(kernel_vector.shape).cuda()
            kernel_pos = torch.max(kernel_vector, kernel_zeros)
            kernel_neg = -1 * torch.min(kernel_vector, kernel_zeros)
            kernel_pos = torch.min(defect_pos_0, kernel_pos)
            kernel_pos = torch.max(defect_pos_1, kernel_pos)
            kernel_neg = torch.min(defect_neg_0, kernel_neg)
            kernel_neg = torch.max(defect_neg_1, kernel_neg)
            kernel_vector = kernel_pos - kernel_neg
            weight = kernel_vector.reshape(weight.shape)
        return weight

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Variation(Function):
    @staticmethod
    def forward(self, weight):
        Macroparameters = MacroParam(Dir="Parameters")
        Specparameters = SpecParam(Dir="Parameters")
        parallism = Specparameters['Subarray'][0]

        NVMSTD = Macroparameters['NVMSTD']
        # print(weight.size())
        if len(weight.size()) == 4:
            [Cout,Cin,k,k] = weight.size()
            coeff = math.sqrt(math.pow(2,(k*k*Cin-1) / parallism))
        if len(weight.size()) == 2:
            [Cout, Cin] = weight.size()
            coeff = math.sqrt(math.pow(2, (Cin - 1) / parallism))


        weight = random.gauss(weight, 0.01*NVMSTD/coeff )
        return weight

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input



class Nonlinear(Function):
    @staticmethod
    def forward(self, input):
        Macroparameters = MacroParam(Dir="Parameters")


        Sign_input = torch.sign(input)
        input = float(Macroparameters['nonlinear_coeff1']) + input * float(Macroparameters['nonlinear_coeff2']) + torch.pow((input), 2) * Sign_input * float(Macroparameters['nonlinear_coeff3'])
        return input

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input



class Hardware_Defect(nn.Module):
    def __init__(self,layer,Dir="Parameters"):
        super(Hardware_Defect, self).__init__()
        self.layer = layer
        self.Dir = Dir
        
    def f_defect(self,weight):
        numbers = []
        with open(os.path.join(app_path(), self.Dir, "layer.txt"),"r") as f:
            for line in f:
                num = int(line.strip())
                numbers.append(num)
        total_layer = numbers[1]
        # [conv_layer,total_layer] = torch.load('./Parameters/layer.json')
        [total_epoch, epoch] = torch.load(os.path.join(app_path(), self.Dir, "epoch.json"))
        # print(f"total_epoch = {total_epoch}, epoch = {epoch}")
        insert_layer = round(epoch/total_epoch*total_layer)
        if self.layer <= insert_layer: # TODO: why this condition?
            global filename
            filename = app_path() + '/defect/Vgg_l' + str(self.layer) + '.pth'
            # print("Apply defect!!!")
            weight = Defect.apply(weight)

        return weight

    def forward(self,weight):
        weight = self.f_defect(weight)
        return weight

class Hardware_Variation(nn.Module):
    def __init__(self,layer,Dir="Parameters"):
        super(Hardware_Variation, self).__init__()
        self.layer = layer
        self.Dir = Dir
    def f_variation(self,weight):
        Macroparameters = MacroParam(Dir=self.Dir)

        [total_epoch, epoch] = torch.load(os.path.join(app_path(), self.Dir, "epoch.json"))
        numbers = []
        with open(os.path.join(app_path(), self.Dir, "layer.txt"),"r") as f:
            for line in f:
                num = int(line.strip())
                numbers.append(num)
        total_layer = numbers[1]
        # [conv_layers,total_layer] = torch.load(os.path.join(app_path(), 'Parameters/layer.json'))
        insert_layer = round(epoch / total_epoch * total_layer)
        global NVMSTD
        NVMSTD = Macroparameters['NVMSTD'] / (
                Macroparameters['MaxConductance'] / Macroparameters['MinConductance']) * math.sqrt(
            (math.pow(2, 2 * Macroparameters['Weight_precision']) - 1) * (
                    math.pow(2, Macroparameters['NVM_states']) - 1) / (
                    math.pow(2, Macroparameters['NVM_states']) + 1))
        if self.layer <= insert_layer:
            # print("Applying Variation!!!")
            weight = Variation.apply(weight)
        return weight
    def forward(self,weight):
        weight = self.f_variation(weight)
        return weight

class Hardware_Nonlinear(nn.Module):
    def __init__(self):
        super(Hardware_Nonlinear, self).__init__()

    def f_nonlinear(self,input):
        coeff_input = Nonlinear.apply(input)
        return coeff_input
    def forward(self,input):
        coeff_input = self.f_nonlinear(input)
        return coeff_input




# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class Conv2d_Q(nn.Conv2d):
    Macroparameters = MacroParam()

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            layer,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            Dir="Parameters",
            a_bits=Macroparameters['DAC_resolution'],
            w_bits=Macroparameters['Weight_precision'],
            o_bits=Macroparameters['ADC_resolution'],
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,


        )
        # 实例化调用A和W量化器
        self.layer = layer
        self.Dir = Dir
        
        if isinstance(w_bits, int):
            self.w_bits = w_bits
        else:
            self.w_bits = w_bits[self.layer - 1]
        if isinstance(a_bits, int):
            self.a_bits = a_bits
        else:
            self.a_bits = a_bits[self.layer - 1]

        Trainfactors = trainParam(Dir=self.Dir)
        self.quan = Trainfactors['quantization']
        self.retrain = Trainfactors['retrain']
        self.nonlinear = Trainfactors['nonlinear']
        self.variation = Trainfactors['variation']
        self.defect = Trainfactors['defect']

        self.activation_quantizer = activation_quantize(a_bits=self.a_bits)
        self.weight_quantizer = weight_quantize(w_bits=self.w_bits)
        self.Hardware_defect = Hardware_Defect(self.layer,Dir=self.Dir)
        self.Hardware_variation = Hardware_Variation(self.layer,Dir=self.Dir)
        self.Hardware_vmm = HardwareVMMoperation(self.Dir)
        self.Hardware_nonlinear = Hardware_Nonlinear()
        
        numbers = []
        with open(os.path.join(app_path(), "Parameters/layer.txt"),"r") as f:
            for line in f:
                num = int(line.strip())
                numbers.append(num)
        conv_layers = numbers[0]
        # [conv_layers,total_layers] = torch.load(os.path.join(app_path(), 'Parameters/layer.json'))
        if self.layer > conv_layers:
            with open(os.path.join(app_path(), "Parameters/layer.txt"),"w") as f:
                f.write(str(self.layer) + "\n")
                f.write(str(self.layer) + "\n")
            # torch.save([self.layer,self.layer], os.path.join(app_path(), 'Parameters/layer.json'))

    def forward(self, input):
        # print(f'[INFO](retrain_modules.py)input.shape = {input.shape}')

        weight = self.weight
        if self.retrain == True or self.training == False:
            if self.quan == True:
                input = self.activation_quantizer(input)
                weight = self.weight_quantizer(weight)
            # Actually, defect,variation and nonlinear are only applied in retrain module.
            if self.defect == True:
                weight = self.Hardware_defect(weight)
            if self.variation == True:
                weight = self.Hardware_variation(weight)
            if self.nonlinear == True:
                input = self.Hardware_nonlinear(input)

        optparam = OptParam(Dir=self.Dir)
        if optparam['evaluate_mode'] == True:
            output = F.conv2d(
                input=input,
                weight=weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            
            # Write the self/weight/input relevant data into files here
            total_spike = 0
            for bit in range(int(self.a_bits)):
                input_spike = input % 2
                input = torch.floor(input / 2)
                total_spike += torch.sum(input_spike)
            input_sparsity = total_spike/self.a_bits/torch.sum(torch.ones(input.size()))
            with open(os.path.join(app_path(), "src/data_transmiss/PyInput.txt"), 
                      'w', encoding="utf-8") as f:
                f.write(f'shape: {input.size()[0]},{input.size()[1]},{input.size()[2]},{input.size()[3]}\n')
                f.write(f'input_sparsity: {input_sparsity}\n')

            with open(os.path.join(app_path(), "src/data_transmiss/PyParam.txt"), 
                      'w', encoding="utf-8") as f:
                f.write(f'layer: {self.layer}\n')
                f.write(f'stride: {self.stride[0]},{self.stride[1]}\n')
                f.write(f'kernel_size: {self.kernel_size[0]},{self.kernel_size[1]}\n')
                f.write(f'padding: {self.padding[0]},{self.padding[1]}\n')
                f.write(f'w_precision: {self.w_bits}\n')
                f.write(f'a_precision: {self.a_bits}\n')
            
            scale = float(2 ** self.w_bits - 1)
            # weight_sparsity = torch.sum(abs(weight))/torch.sum(torch.ones(weight.size())) / 2 
            weight_sparsity = torch.sum(abs(weight))/torch.sum(torch.ones(weight.size())*scale) / 2 
            with open(os.path.join(app_path(), "src/data_transmiss/PyWeight.txt"), 
                      'w', encoding="utf-8") as f:
                f.write(f'shape: {weight.size()[0]},{weight.size()[1]},{weight.size()[2]},{weight.size()[3]}\n')
                f.write(f'weight_sparsity: {weight_sparsity}\n')
            # print(f'shape: {weight.size()[0]},{weight.size()[1]},{weight.size()[2]},{weight.size()[3]}')
            curDir = os.getcwd()
            os.chdir("./refactor/build")
            os.system("./main --mapping_modules " + "user_name " + str(1)) # any user_name/tid is ok here.
            os.chdir(curDir)
            # mapping_modules(self, weight, input)
        else:

            # if self.quan == True:
            #     output = self.Hardware_vmm.Hard_conv(a_bits=self.a_bits, layer=self.layer, training=self.training,
            #                                          input=input, kernel=weight, bias=self.bias,
            #                                          padding=self.padding, stride=self.stride)
            # else:
                output = F.conv2d(
                    input=input,
                    weight=weight,
                    bias=self.bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )

        return output


# ********************* 量化全连接（同时量化A/W，并做全连接） ***********************
class Linear_Q(nn.Linear):
    Macroparameters = MacroParam()

    def __init__(self, in_features, out_features, layer, bias=False,a_bits=Macroparameters['DAC_resolution'],
            w_bits=Macroparameters['Weight_precision'],o_bits=Macroparameters['ADC_resolution'],Dir="Parameters"):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.layer = layer
        self.Dir = Dir
        
        if isinstance(w_bits, int):
            self.w_bits = w_bits
        else:
            self.w_bits = w_bits[self.layer - 1]
        if isinstance(a_bits, int):
            self.a_bits = a_bits
        else:
            self.a_bits = a_bits[self.layer - 1]
        # self.o_bits = o_bits
        # self.o_scale = float(2 ** o_bits - 1)
        Trainfactors = trainParam(Dir=self.Dir)
        self.quan = Trainfactors['quantization']
        self.retrain = Trainfactors['retrain']
        self.nonlinear = Trainfactors['nonlinear']
        self.variation = Trainfactors['variation']
        self.defect = Trainfactors['defect']

        self.activation_quantizer = activation_quantize(a_bits=self.a_bits)
        self.weight_quantizer = weight_quantize(w_bits=self.w_bits)
        self.Hardware_defect = Hardware_Defect(self.layer,Dir=self.Dir)
        self.Hardware_variation = Hardware_Variation(self.layer,Dir=self.Dir)
        self.Hardware_vmm = HardwareVMMoperation(self.Dir)

        self.Hardware_nonlinear = Hardware_Nonlinear()
        numbers = []
        with open(os.path.join(app_path(), "Parameters/layer.txt"),"r") as f:
            for line in f:
                num = int(line.strip())
                numbers.append(num)
        conv_layers = numbers[0]
        total_layers = numbers[1]
        # [conv_layers, total_layers] = torch.load(os.path.join(app_path(), 'Parameters/layer.json'))
        if self.layer > total_layers:
            with open(os.path.join(app_path(), "Parameters/layer.txt"),"w") as f:
                f.write(str(conv_layers) + "\n")
                f.write(str(self.layer) + "\n")
                # print(f'In Linear_Q, self.layer = {self.layer}, conv_layers = {conv_layers}')
            # torch.save([conv_layers, self.layer], os.path.join(app_path(), 'Parameters/layer.json'))

        # self.first_layer = first_layer
    def forward(self, input):

        weight = self.weight
        if self.retrain == True or self.training == False:
            if self.quan == True:
                input = self.activation_quantizer(input)
                weight = self.weight_quantizer(weight)
            if self.defect == True:
                weight = self.Hardware_defect(weight)
            if self.variation == True:
                weight = self.Hardware_variation(weight)
            if self.nonlinear == True:
                input = self.Hardware_nonlinear(input)

        optparam = OptParam(Dir=self.Dir)
        if optparam['evaluate_mode'] == True:
            output = F.linear(
                input=input,
                weight=weight,
                bias=self.bias,
            )
            
            total_spike = 0
            for bit in range(int(self.a_bits)):
                input_spike = input % 2
                input = torch.floor(input / 2)
                total_spike += torch.sum(input_spike)
            input_sparsity = total_spike/self.a_bits/torch.sum(torch.ones(input.size()))
            with open(os.path.join(app_path(), "src/data_transmiss/PyInput.txt"), 
                      'w', encoding="utf-8") as f:
                f.write(f'shape: {input.size()[0]},{input.size()[1]}\n')
                f.write(f'input_sparsity: {input_sparsity}\n')

            with open(os.path.join(app_path(), "src/data_transmiss/PyParam.txt"), 
                      'w', encoding="utf-8") as f:
                f.write(f'layer: {self.layer}\n')
                f.write(f'stride: {1},{1}\n')
                f.write(f'kernel_size: {0},{0}\n') # Set kernel_size to [0,0] in Linear
                f.write(f'padding: {0},{0}\n') # Set padding to [0,0] in Linear
                f.write(f'w_precision: {self.w_bits}\n')
                f.write(f'a_precision: {self.a_bits}\n')

            scale = float(2 ** self.w_bits - 1)
            weight_sparsity = torch.sum(abs(weight))/torch.sum(torch.ones(weight.size())*scale) / 2 
            with open(os.path.join(app_path(), "src/data_transmiss/PyWeight.txt"), 
                      'w', encoding="utf-8") as f:
                f.write(f'shape: {weight.size()[0]},{weight.size()[1]}\n')
                f.write(f'weight_sparsity: {weight_sparsity}\n')
            curDir = os.getcwd()
            os.chdir("./refactor/build")
            os.system("./main --mapping_modules " + "user_name " + str(1)) # any tid is ok here.
            os.chdir(curDir)
            # mapping_modules(self, weight, input)
        else:
            # if self.quan == True:
            #       output = self.Hardware_vmm.Hard_fc(a_bits=self.a_bits, layer=self.layer, training=self.training,
            #                                    input=input, weight=weight, bias=self.bias)
            # else:
                output = F.linear(
                    input=input,
                    weight=weight,
                    bias=self.bias,
                )

        return output

class MatMul(nn.Module):
    """
    Specially Designed for STGCN, where constant params in all matmuls have two
    dimensions. einsum can also be converted into matmul.
    """
    def __init__(self, layer, Dir = "Parameters") -> None:
        super().__init__()
        self.Dir = Dir
        self.layer = layer
        Macroparameters = MacroParam(Dir = self.Dir)
        self.w_bits = Macroparameters['Weight_precision']
        self.a_bits = Macroparameters['DAC_resolution']

        Trainfactors = trainParam(Dir=self.Dir)
        self.quan = Trainfactors['quantization']
        self.retrain = Trainfactors['retrain']
        
        self.activation_quantizer = activation_quantize(a_bits=self.a_bits)
        self.weight_quantizer = weight_quantize(w_bits=self.w_bits)
        
    # constant = 0, M1 = constant; constant = 1, M2 = constant
    # TODO: only quantization is considered here.
    def forward(self, M1, M2, constant=0):
        if self.retrain == True or self.training == False:
            if self.quan == True:
                if constant == 0:
                    M2 = self.activation_quantizer(M2)
                    M1 = self.weight_quantizer(M1)
                else:
                    M1 = self.activation_quantizer(M1)
                    M2 = self.weight_quantizer(M2)
        
        optparam = OptParam(Dir=self.Dir)
        if optparam['evaluate_mode'] == True:
            if constant == 0:
                weight_sparsity = torch.sum(abs(M1))/torch.sum(torch.ones(M1.size())) / 2 
                with open(os.path.join(app_path(), "src/data_transmiss/PyWeight.txt"), 
                        'w', encoding="utf-8") as f:
                    f.write(f'shape: {M1.size()[0]},{M1.size()[1]}\n')
                    f.write(f'weight_sparsity: {weight_sparsity}\n')
                    f.write(f'type: LW\n')
                total_spike = 0
                for bit in range(int(self.a_bits)):
                    input_spike = M2 % 2
                    M2 = torch.floor(M2 / 2)
                    total_spike += torch.sum(input_spike)
                input_sparsity = total_spike/self.a_bits/torch.sum(torch.ones(M2.size()))
                with open(os.path.join(app_path(), "src/data_transmiss/PyInput.txt"), 
                        'w', encoding="utf-8") as f:
                    ch = ''
                    f.write("shape: ")
                    for i, size in enumerate(M2.size()):
                        f.write(f"{ch}{size}")
                        ch = ','
                    f.write(f'\ninput_sparsity: {input_sparsity}\n') 
            else:
                weight_sparsity = torch.sum(abs(M2))/torch.sum(torch.ones(M2.size())) / 2 
                with open(os.path.join(app_path(), "src/data_transmiss/PyWeight.txt"), 
                        'w', encoding="utf-8") as f:
                    f.write(f'shape: {M2.size()[0]},{M2.size()[1]}\n')
                    f.write(f'weight_sparsity: {weight_sparsity}\n')
                    f.write(f'type: RW\n')
                total_spike = 0
                for bit in range(int(self.a_bits)):
                    input_spike = M1 % 2
                    M1 = torch.floor(M1 / 2)
                    total_spike += torch.sum(input_spike)
                input_sparsity = total_spike/self.a_bits/torch.sum(torch.ones(M1.size()))
                with open(os.path.join(app_path(), "src/data_transmiss/PyInput.txt"), 
                        'w', encoding="utf-8") as f:
                    ch = ''
                    f.write("shape: ")
                    for i, size in enumerate(M1.size()):
                        f.write(f"{ch}{size}")
                        ch = ','
                    f.write(f'\ninput_sparsity: {input_sparsity}\n') 
            
            with open(os.path.join(app_path(), "src/data_transmiss/PyParam.txt"), 
                      'w', encoding="utf-8") as f:
                f.write(f'layer: {self.layer}\n') # matmul counted in layers??
                f.write(f'stride: {1},{1}\n')
                f.write(f'kernel_size: {0},{0}\n') # Set kernel_size to [0,0] in Linear
                f.write(f'padding: {0},{0}\n') # Set padding to [0,0] in Linear
                f.write(f'w_precision: {self.w_bits}\n')
                f.write(f'a_precision: {self.a_bits}\n')
        
            curDir = os.getcwd()
            os.chdir("./refactor/build")
            os.system("./main --mapping_modules " + "user_name " + str(1)) # any tid is ok here.
            os.chdir(curDir)  
        
        output = torch.matmul(M1, M2)
        return output
        

class Relu_S(nn.ReLU):
    def __init__(self,inplace, layer,Dir="Parameters"):
        super(Relu_S, self).__init__()
        self.inplace = inplace
        self.layer = layer
        self.Dir = Dir

    def forward(self, input):
        output = F.relu(input,self.inplace)
        optparam = OptParam(Dir=self.Dir)
        if optparam['evaluate_mode'] == True:
            activation_mode = 0
            with open(os.path.join(app_path(), "src/data_transmiss/PyActInput.txt"), 
                      'w', encoding="utf-8") as f:
                if len(input.size()) == 4:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]},{input.size()[2]},{input.size()[3]}\n')
                else:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]}\n')
                f.write(f'mode: {activation_mode}\n')
            curDir = os.getcwd()
            os.chdir("./refactor/build")
            os.system("./main --activation_modules " + "user_name " + str(1)) # any tid is ok here.
            os.chdir(curDir)
            # activation_modules(self, input, activation_mode)
            # if(self.layer == 1):
            #     print("In Relu_S, Layer == 1")
            # if(self.layer == 9):
            #     print("In Relu_S, Layer == 9")

        return output

class Sigmoid_S(nn.Sigmoid):
    def __init__(self,inplace, layer,Dir="Parameters"):
        super(Sigmoid_S, self).__init__()
        self.inplace = inplace
        self.layer = layer
        self.Dir = Dir

    def forward(self, input):
        output = torch.sigmoid(input)
        optparam = OptParam(Dir=self.Dir)
        if optparam['evaluate_mode'] == True:
            activation_mode = 2
            with open(os.path.join(app_path(), "src/data_transmiss/PyActInput.txt"), 
                      'w', encoding="utf-8") as f:
                if len(input.size()) == 4:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]},{input.size()[2]},{input.size()[3]}\n')
                else:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]}\n')
                f.write(f'mode: {activation_mode}\n')
            curDir = os.getcwd()
            os.chdir("./refactor/build")
            os.system("./main --activation_modules " + "user_name " + str(1)) # any tid is ok here.
            os.chdir(curDir)
            # activation_modules(self, input, activation_mode)
            # if(self.layer == 1):
            #     print("In Sigmoid_S, Layer == 1")
            # if(self.layer == 9):
            #     print("In Sigmoid_S, Layer == 9")

        return output


class Maxpool_S(nn.MaxPool2d):
    def __init__(self,kernel_size, stride, layer, padding=0,Dir="Parameters"):
        super(Maxpool_S, self).__init__(kernel_size=kernel_size,stride=stride,padding=padding)
        self.kernel_size= kernel_size
        self.stride = stride
        self.padding = padding
        self.layer = layer
        self.Dir = Dir

    def forward(self,input):
        output = F.max_pool2d(input,self.kernel_size,stride=self.stride,padding=self.padding)
        optparam = OptParam(Dir=self.Dir)
        if optparam['evaluate_mode'] == True:
            activation_mode = 1
            with open(os.path.join(app_path(), "src/data_transmiss/PyActInput.txt"), 
                      'w', encoding="utf-8") as f:
                if len(input.size()) == 4:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]},{input.size()[2]},{input.size()[3]}\n')
                else:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]}\n')
                f.write(f'mode: {activation_mode}\n')
            curDir = os.getcwd()
            os.chdir("./refactor/build")
            os.system("./main --activation_modules " + "user_name " + str(1)) # any tid is ok here.
            os.chdir(curDir)
            # activation_modules(self, input, activation_mode)
            # if(self.layer == 1):
            #     print("In Maxpool_S, Layer == 1")
            # if(self.layer == 9):
            #     print("In Maxpool_S, Layer == 9")

        return output

class Avgpool_S(nn.AvgPool2d):
    def __init__(self,kernel_size, stride, layer, padding=0,Dir="Parameters"):
        super(Avgpool_S, self).__init__(kernel_size=kernel_size,stride=stride,padding=padding)
        self.kernel_size= kernel_size
        self.stride = stride
        self.padding = padding
        self.layer = layer
        self.Dir = Dir

    def forward(self,input):
        output = F.avg_pool2d(input,self.kernel_size,stride=self.stride,padding=self.padding)
        optparam = OptParam(Dir=self.Dir)
        if optparam['evaluate_mode'] == True:
            activation_mode = 1
            with open(os.path.join(app_path(), "src/data_transmiss/PyActInput.txt"), 
                      'w', encoding="utf-8") as f:
                if len(input.size()) == 4:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]},{input.size()[2]},{input.size()[3]}\n')
                else:
                    f.write(f'shape: {input.size()[0]},{input.size()[1]}\n')
                f.write(f'mode: {activation_mode}\n')
            curDir = os.getcwd()
            os.chdir("./refactor/build")
            os.system("./main --activation_modules " + "user_name " + str(1)) # any tid is ok here.
            os.chdir(curDir)
            # activation_modules(self, input, activation_mode)
            # if(self.layer == 1):
            #     print("In Avgpool_S, Layer == 1")
            # if(self.layer == 9):
            #     print("In Avgpool_S, Layer == 9")

        return output

class Sequential_S(nn.Sequential):

    def __init__(self,*args,layer,Dir="Parameters"):
        super(Sequential_S,self).__init__(*args)
        self.layer = layer
        self.Dir = Dir

    def forward(self, input):
        # print("Into Sequential_S!!!")
        # print(f"self.layer = {self.layer}")
        init_layer = self.layer
        for module in self:

            if isinstance(module,Conv2d_Q) and self.layer > 1:
                if module.kernel_size[0] == 3 and module.in_channels != module.groups:
                    # print(f'layer = {self.layer}, input shape = {input.size()}')
                    layer_connect('layerconnect', self.layer - 1, self.layer, 'conv3*3', input.size(),Dir=self.Dir)
                if module.kernel_size[0] == 3 and module.in_channels == module.groups:
                    layer_connect('layerconnect', self.layer - 1, self.layer, 'dw_conv3*3', input.size(),Dir=self.Dir)
                if module.kernel_size[0] == 1:
                    layer_connect('layerconnect', self.layer - 1, self.layer, 'conv1*1', input.size(),Dir=self.Dir)


            if isinstance(module, Linear_Q):
                # print(f'layer = {self.layer}, input shape = {input.size()}')
                layer_connect('layerconnect', self.layer - 1, self.layer, 'fc', input.size(),Dir=self.Dir)

            if isinstance(module,Conv2d_Q) or isinstance(module,Linear_Q):
                self.layer += 1

            input = module(input)
        self.layer = init_layer
        return input
