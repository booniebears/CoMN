import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from Parameters import *
from torch.autograd import Function




class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input



#********************************** Partial sums activation ****************************************
class output_quantize(nn.Module):
    def __init__(self, o_bits,a_scale, w_scale,Dir="Parameters"):
        super().__init__()
        self.Dir = Dir
        Trainfactors = trainParam(Dir=self.Dir)
        self.o_bits = o_bits
        self.quan = Trainfactors['quantization']
        self.retrain = Trainfactors['retrain']
        self.w_scale = w_scale
        self.a_scale = a_scale


    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input,layer):

        Specparameters = SpecParam(Dir=self.Dir)
        parallism = Specparameters['Subarray'][0]
        if isinstance(self.o_bits,int):
            o_bits = self.o_bits
        else:
            o_bits = self.o_bits[layer-1]
        if o_bits == 32:
            output = input

        elif o_bits == 1:
            print('! Binary quantization is not supported !')
            assert o_bits != 1
        else:
            scale = float(2 ** o_bits - 1)

            # max_input = max(input.reshape(-1))
            # print(max_input)
            maxcurrent = 0.02*parallism
            if isinstance(self.w_scale,float) and self.w_scale < 2**32-1:
                maxcurrent *= self.w_scale
            # else:
            #     maxcurrent *= self.w_scale[layer-1]
            if isinstance(self.a_scale,float) and self.a_scale < 2**32-1:
                maxcurrent *= self.a_scale
            # else:
            #     maxcurrent *= self.a_scale[layer-1]
            # print(torch.max(input))
            # print(maxcurrent)
            output = torch.clamp(input, -1 * maxcurrent, maxcurrent)
            # max_out = torch.ones(output.size())*maxcurrent
            # sum_out = torch.load("temp_sum_out.json")
            # sum_out += torch.sum(max_out)
            # torch.save(sum_out,"temp_sum_out.json")
            # output = input
            # output = 1.0698*input - 0.01948*input*input
            # output = 1.043 * input - 0.0191 * input * input
            # output = 1 * input - 0.01777 * input * input
            # output = 1.4*input - 0.133*input*input
            # output = 1.127 * input - 0.1127 * input * input
            # output = 1 * input - 0.09497 * input * input
            if self.retrain == True or self.training == False:
                if self.quan == True:
                    output = output / maxcurrent * scale
                    output = self.round(output)
                    output = output/scale*maxcurrent

        return output


class HardwareVMMoperation(nn.Module):
    def __init__(self,Dir="Parameters"):
        super().__init__()
        self.Dir = Dir
        Macroparameters = MacroParam(Dir=self.Dir)

        self.a_scale = float(2 ** Macroparameters['DAC_resolution'] - 1)
        self.w_scale = float(2 ** Macroparameters['Weight_precision'] - 1)
        self.Partialquantizer = output_quantize(o_bits=Macroparameters['ADC_resolution'],a_scale = self.a_scale,w_scale = self.w_scale,
                                                Dir = self.Dir)
        # self.o_bits = Macroparameters['ADC_resolution']
        # self.a_bits = Macroparameters['DAC_resolution']


    def round(self, input):
        output = Round.apply(input)
        return output

    def Hard_conv(self,a_bits,layer, training, input, kernel, bias, padding, stride):

        Specparameters = SpecParam(Dir=self.Dir)
        parallism = Specparameters['Subarray'][0]
        self.training = training
        # if isinstance(padding, tuple):
        #     padding = torch.tensor(padding[0])
        input = F.pad(input, padding + padding)

        batch_size = input.shape[0]
        input_h, input_w = input.shape[2:4]
        kernel_h, kernel_w = kernel.shape[2:4]
        out_channel, in_channel = kernel.shape[0:2]
        output_h = math.floor((input_h - kernel_h) / stride[1] + 1)
        output_w = math.floor((input_w - kernel_w) / stride[0] + 1)

        unfold = nn.Unfold(kernel_size=(kernel_h, kernel_w), stride=stride)
        input_vector = unfold(input)

        kernel_vector = kernel.reshape(kernel.shape[0], -1).T
        input_vector = input_vector.permute(0, 2, 1).contiguous()

        total_row = kernel_vector.shape[0]
        num = math.ceil(total_row / parallism)
        output = 0
        for i in range(num):
            if (i + 1) * parallism <= total_row:
                out = input_vector[:, :, i * parallism:(i + 1) * parallism] @ kernel_vector[i * parallism:(i + 1) * parallism, :]
                out = self.Partialquantizer(out,layer)
                output += out
            else:

                out = input_vector[:, :, i * parallism:total_row] @ kernel_vector[i * parallism:total_row, :]
                out = self.Partialquantizer(out,layer)
                output += out


        if bias != None:

            output += bias
        # output = (input_vector @ kernel_vector) + param.bias
        output = output.reshape(batch_size, output_h, output_w, out_channel).permute(0, 3, 1, 2).contiguous()

        return output

    def Hard_fc(self,a_bits,layer,training, input, weight, bias):
        Specparameters = SpecParam(Dir=self.Dir)
        parallism = Specparameters['Subarray'][0]
        self.training = training
        # print(layer)
        weight = weight.T
        batch_size = input.shape[0]
        input_vector = input.reshape(batch_size, input.shape[1])

        total_row = weight.shape[0]
        num = math.ceil(total_row / parallism)
        output = 0
        for i in range(num):
            if (i + 1) * parallism <= total_row:

                out = input_vector[:, i * parallism:(i + 1) * parallism] @ weight[i * parallism:(i + 1) * parallism, :]
                out = self.Partialquantizer(out,layer)
                output += out
            else:

                out = input_vector[:, i * parallism:total_row] @ weight[i * parallism:total_row, :]
                out = self.Partialquantizer(out,layer)
                output += out


        if bias != None:
            output += bias
        # output = (input_vector @ weight) + bias
        output = output.reshape(batch_size, -1)

        return output



# class HardwareVMMoperation(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Partialquantizer = output_quantize(o_bits=Macroparameters['ADC_resolution'])
#         self.o_bits = Macroparameters['ADC_resolution']
#         self.a_bits = Macroparameters['DAC_resolution']
#         self.a_scale = float(2 ** Macroparameters['DAC_resolution'] - 1)
#
#     def round(self, input):
#         output = Round.apply(input)
#         return output
#
#     def Hard_conv(self,training, input, kernel, bias, padding, stride):
#         self.training = training
#         input = F.pad(input, padding + padding)
#
#         batch_size = input.shape[0]
#         input_h, input_w = input.shape[2:4]
#         kernel_h, kernel_w = kernel.shape[2:4]
#         out_channel, in_channel = kernel.shape[0:2]
#         output_h = math.floor((input_h - kernel_h) / stride[1] + 1)
#         output_w = math.floor((input_w - kernel_w) / stride[0] + 1)
#
#         unfold = nn.Unfold(kernel_size=(kernel_h, kernel_w), stride=stride)
#         input_vector = unfold(input)
#
#         kernel_vector = kernel.reshape(kernel.shape[0], -1).T
#         input_vector = input_vector.permute(0, 2, 1).contiguous()
#         temp_input = torch.ones(input_vector.size())
#         # temp_kernel = torch.ones(kernel_vector.size())
#         sum_out = torch.load("temp_sum_out.json")
#         sum_out = max(sum_out,torch.max(input_vector))
#         # sum_out += torch.sum(abs(temp_input))
#         torch.save(sum_out,"temp_sum_out.json")
#
#         kernel_zeros = torch.zeros(kernel_vector.shape).cuda()
#         kernel_pos = torch.max(kernel_vector, kernel_zeros)
#         kernel_neg = -1 * torch.min(kernel_vector, kernel_zeros)
#         input_quan= (input_vector*self.a_scale).type(torch.int8)
#
#         total_row = kernel_vector.shape[0]
#         num = math.ceil(total_row / parallism)
#         Output = 0
#         for j in range(self.a_bits):
#             input_pulse = input_quan%2
#
#             input_quan = torch.floor(input_quan/2)
#             output = 0
#             for i in range(num):
#                 if (i + 1) * parallism <= total_row:
#                     out_pos = input_pulse[:, :, i * parallism:(i + 1) * parallism] @ kernel_pos[
#                                                                                      i * parallism:(i + 1) * parallism,
#                                                                                      :]
#                     # out_pos = torch.clamp(out_pos, -1 * maxcurrent, maxcurrent)
#                     out_pos = self.Partialquantizer(out_pos)
#                     out_neg = input_pulse[:, :, i * parallism:(i + 1) * parallism] @ kernel_neg[
#                                                                                      i * parallism:(i + 1) * parallism,
#                                                                                      :]
#                     # out_neg = torch.clamp(out_neg, -1 * maxcurrent, maxcurrent)
#                     out_neg = self.Partialquantizer(out_neg)
#                     out = out_pos - out_neg
#
#                     # out = input_vector[:, :, i * parallism:(i + 1) * parallism] @ kernel_vector[i * parallism:(i + 1) * parallism, :]
#                     # out = torch.clamp(out, -1 * maxcurrent, maxcurrent)
#                     output += out
#                 else:
#                     out_pos = input_pulse[:, :, i * parallism:total_row] @ kernel_pos[i * parallism:total_row, :]
#                     # out_pos = torch.clamp(out_pos, -1 * maxcurrent, maxcurrent)
#                     out_pos = self.Partialquantizer(out_pos)
#                     out_neg = input_pulse[:, :, i * parallism:total_row] @ kernel_neg[i * parallism:total_row, :]
#                     # out_neg = torch.clamp(out_neg, -1 * maxcurrent, maxcurrent)
#                     out_neg = self.Partialquantizer(out_neg)
#                     out = out_pos - out_neg
#                     # out = input_vector[:, :, i * parallism:total_row] @ kernel_vector[i * parallism:total_row, :]
#                     # out = torch.clamp(out, -1 * maxcurrent, maxcurrent)
#                     output += out
#                 # sum_out = torch.load("temp_sum_out.json")
#                 # sum_out += torch.sum(abs(out))
#                 # torch.save(sum_out,"temp_sum_out.json")
#             # output = output / (maxcurrent) * (2**self.o_bits-1)
#             # output = self.round(output)
#             # output = output * (maxcurrent) / (2 ** self.o_bits - 1)
#
#             Output += output*2^j
#
#
#
#         if bias != None:
#             Output += bias
#         # output = (input_vector @ kernel_vector) + param.bias
#         output = Output.reshape(batch_size, output_h, output_w, out_channel).permute(0, 3, 1, 2).contiguous()
#
#         return output
#
#     def Hard_fc(self,training, input, weight, bias):
#         self.training = training
#         weight = weight.T
#         batch_size = input.shape[0]
#         input_vector = input.reshape(batch_size, input.shape[1])
#         temp_input = torch.ones(input_vector.size())
#         # temp_weight = torch.ones(weight.size())
#         sum_out = torch.load("temp_sum_out.json")
#         sum_out = max(sum_out,torch.max(input_vector))
#         # sum_out += torch.sum(abs(temp_input))
#         torch.save(sum_out, "temp_sum_out.json")
#         weight_zeros = torch.zeros(weight.shape).cuda()
#         weight_pos = torch.max(weight, weight_zeros)
#         weight_neg = -1 * torch.min(weight, weight_zeros)
#         input_quan = (input_vector * self.a_scale).type(torch.int8)
#         total_row = weight.shape[0]
#         num = math.ceil(total_row / parallism)
#         Output = 0
#         for j in range(self.a_bits):
#             input_pulse = input_quan % 2
#             input_quan = torch.floor(input_quan / 2)
#             output = 0
#             for i in range(num):
#                 if (i + 1) * parallism <= total_row:
#                     out_pos = input_pulse[:, i * parallism:(i + 1) * parallism] @ weight_pos[
#                                                                                   i * parallism:(i + 1) * parallism, :]
#                     # out_pos = torch.clamp(out_pos, -1 * maxcurrent, maxcurrent)
#                     out_pos = self.Partialquantizer(out_pos)
#                     out_neg = input_pulse[:, i * parallism:(i + 1) * parallism] @ weight_neg[
#                                                                                   i * parallism:(i + 1) * parallism, :]
#                     # out_neg = torch.clamp(out_neg, -1 * maxcurrent, maxcurrent)
#                     out_neg = self.Partialquantizer(out_neg)
#                     out = out_pos - out_neg
#
#                     # out = input_vector[:, i * parallism:(i + 1) * parallism] @ weight[i * parallism:(i + 1) * parallism, :]
#                     # out = self.Partialquantizer(out)
#                     output += out
#                 else:
#                     out_pos = input_pulse[:, i * parallism:total_row] @ weight_pos[i * parallism:total_row, :]
#                     # out_pos = torch.clamp(out_pos, -1 * maxcurrent, maxcurrent)
#                     out_pos = self.Partialquantizer(out_pos)
#                     out_neg = input_pulse[:, i * parallism:total_row] @ weight_neg[i * parallism:total_row, :]
#                     # out_neg = torch.clamp(out_neg, -1 * maxcurrent, maxcurrent)
#                     out_neg = self.Partialquantizer(out_neg)
#                     out = out_pos - out_neg
#                     # out = input_vector[:, i * parallism:total_row] @ weight[i * parallism:total_row, :]
#                     # out = self.Partialquantizer(out)
#                     output += out
#                 # sum_out = torch.load("temp_sum_out.json")
#                 # sum_out += torch.sum(abs(out))
#                 # torch.save(sum_out, "temp_sum_out.json")
#             # output = output / (maxcurrent) * (2 ** self.o_bits - 1)
#             # output = self.round(output)
#             # output = output * (maxcurrent) / (2 ** self.o_bits - 1)
#
#
#         Output += output * 2 ^ j
#         if bias != None:
#             Output += bias
#         # output = (input_vector @ weight) + bias
#         output = Output.reshape(batch_size, -1)
#
#         return output



# def VMM(input, weight):
#     [row,col] = weight.shape
#     [batch,cycle,vector] = input.shape
#     output = torch.zeros(batch, cycle, col)
#
#     for c in range(col):
#         temp2 = torch.zeros(batch,cycle)
#         for b in range(batch):
#             temp1 = 0
#             temp = 0
#             numrow = 0
#             for r in range(row):
#                 numrow += 1
#                 if numrow <= parallism:
#                     temp += weight[r,c]*input[b, :, r]
#                 else:
#                     temp1 += temp
#                     temp = weight[r,c]*input[b, :, r]
#                     numrow = 1
#             temp2[b,:] = temp1
#         output[:, :, c] = temp2
#     return output

