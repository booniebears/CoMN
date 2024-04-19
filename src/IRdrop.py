import torch
import math
from Parameters import *
from frozen_dir import app_path

def Eva_IRdrop(kernel):
    Trainfactors = trainParam(Dir="Parameters")
    Specificationparam = SpecParam(Dir="Parameters")
    Subarray = Specificationparam['Subarray']
    correction = Trainfactors['correction']
    kernel_vector = kernel.reshape(kernel.shape[0], -1).T # (k*k*Cin,Cout)
    num_row = math.ceil(kernel_vector.shape[0] / Subarray[0])
    num_col = math.ceil(kernel_vector.shape[1] / Subarray[1])
    kernel_zeros = torch.zeros(kernel_vector.shape).cuda()
    kernel_pos = torch.max(kernel_vector, kernel_zeros)
    kernel_neg = -1*torch.min(kernel_vector, kernel_zeros)
    for i in range(num_row):
        for j in range(num_col):
            W = kernel_pos[i * Subarray[0]:(i + 1) * Subarray[0], j * Subarray[1]:(j + 1) * Subarray[1]]
            G = WGmapping(W)
            if correction == True:
                [Gcorr, mean_error] = WMC(G)
                [Geqv, Veqv] = IRdrop(Gcorr)
                Geqv = Geqv/mean_error

            if correction == False:
                [Geqv, Veqv] = IRdrop(G)

            W = GWmapping(Geqv, W)
            kernel_pos[i * Subarray[0]:(i + 1) * Subarray[0], j * Subarray[1]:(j + 1) * Subarray[1]] = W
    for i in range(num_row):
        for j in range(num_col):
            W = kernel_neg[i * Subarray[0]:(i + 1) * Subarray[0], j * Subarray[1]:(j + 1) * Subarray[1]]
            G = WGmapping(W)
            if correction == True:
                [Gcorr, mean_error] = WMC(G)
                [Geqv, Veqv] = IRdrop(Gcorr)
                Geqv = Geqv/mean_error

            if correction == False:
                [Geqv, Veqv] = IRdrop(G)

            W = GWmapping(Geqv, W)
            kernel_neg[i * Subarray[0]:(i + 1) * Subarray[0], j * Subarray[1]:(j + 1) * Subarray[1]] = W
    kernel_vector = kernel_pos - kernel_neg
    kernel_vector = kernel_vector.T
    kernel = kernel_vector.reshape(kernel.shape)
    return kernel

def Eva_WMC(checkpoint,tid,user_name):
    layer = 0
    accuracy_file = app_path() + "/generate_data/" + user_name + "/accuracy_out/accuracy_out" + str(tid) + ".txt"
    with open(accuracy_file, 'a+', encoding="utf-8") as f:
        f.write("Correcting Weight with IR Drop!!! It will take quite a few minutes...")
        f.write("\n")
    for key in checkpoint['state_dict'].keys():
        print(f"key = {key}")
        temp = key.split('.')
        # if temp[1] == 'weight':
        if temp[2:3] == ['weight'] and len(checkpoint['state_dict'][key].shape) == 4:
            layer = layer + 1
            print('the {0}th layer is simulating/correcting'.format(layer))
            with open(accuracy_file, 'a+', encoding="utf-8") as f:
                f.write('the {0}th layer is simulating/correcting'.format(layer))
                f.write("\n")
            weight = checkpoint['state_dict'][key]
            weight = Eva_IRdrop(kernel=weight)
            checkpoint['state_dict'][key] = weight
    return checkpoint


def WGmapping(weight):
    Macroparameters = MacroParam(Dir="Parameters")
    MaxConductance = Macroparameters['MaxConductance']
    MinConductance = Macroparameters['MinConductance']
    G = weight / torch.max(weight) * (MaxConductance - MinConductance) + MinConductance
    return G

def GWmapping(G,weight):
    Macroparameters = MacroParam(Dir="Parameters")
    MaxConductance = Macroparameters['MaxConductance']
    MinConductance = Macroparameters['MinConductance']
    weight = (G - MinConductance) / (MaxConductance - MinConductance) * torch.max(weight)
    return weight

def IRdrop(Conductance):
    Macroparameters = MacroParam(Dir="Parameters")
    Rwire_row = Macroparameters['Rwire_row']
    Rwire_col = Macroparameters['Rwire_col']
    [Row, Col] = Conductance.shape

    Avg_Conductance = torch.mean(Conductance)
    Avg_Resistance = 1/Avg_Conductance
    Resistance = 1/Conductance
    Reqv = torch.zeros(Conductance.shape).cuda()
    coeff = torch.zeros(Conductance.shape).cuda()
    for i in range (Row):
        for j in range (Col):
            coeff[i,j] = Resistance[i,j]/Avg_Resistance
            Reqv[i,j] = Resistance[i,j] + ((Row+i)*(Row+1-i)/2*Rwire_row + (2*Col-j+1)*j/2*Rwire_col)*coeff[i,j]
    Geqv = 1/Reqv
    Veqv = Geqv/Conductance

    return Geqv,Veqv


def WMC(G):
    lr = 0.01
    pre_error = 1
    iter = 1000
    [Row, Col] = G.shape
    min_error = 0.0001
    correction_G = torch.zeros(G.shape).cuda()

    inital_G = G
    for k in range(iter):

        [Geqv,Veqv] = IRdrop(G)
        error_G = Geqv / inital_G
        std_error = torch.std(error_G)
        mean_error = torch.mean(error_G)
        if std_error > pre_error:
            break

        pre_error = std_error
        WMC_G = G
        for i in range(Row):
            for j in range(Col):
                correction_G[i, j] = G[i, j] + lr * (mean_error * inital_G[i, j] - Geqv[i, j])

        if std_error < min_error:
            break

        scale = (torch.max(correction_G) - torch.min(correction_G)) / (torch.max(inital_G) - torch.min(inital_G))
        G = (correction_G - torch.min(correction_G)) / scale + torch.min(inital_G)

    return WMC_G,mean_error



