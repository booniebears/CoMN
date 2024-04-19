##################################################################################################
############          Scenario 2: Optimizing IMC chip specification for multiple DNN models   ########################
##################################################################################################

from models import *
import numpy as np
from Parameters import *
import time
from Mapping_optimizer import *
from Bayesian.bayes.bayesian_optimization import BayesianOptimization
import math
import sys
import json
from frozen_dir import app_path
temp = sys.stdout

Spec_user_name = ""
Spec_weight_name = ""
Spec_tid = ""


def specification(Subarray, Macronumbers, buswidthTile, buffersizeTile, ColumnMUX, Meshflitband, Htreeflitband):
    global Spec_User_name
    global Spec_weight_name
    global Spec_tid
    optparam = OptParam()
    arrayrow = int(math.pow(2, round(math.log2(Subarray))))
    arraycol = int(arrayrow)
    Subarray = [arrayrow, arraycol]

    buffersizeTile =  int(math.pow(2, round(math.log2(buffersizeTile))))
    buswidthTile =  int(math.pow(2, round(math.log2(buswidthTile))))
    ColumnMUX =  int(math.pow(2, round(math.log2(ColumnMUX))))
    Macroparameters = MacroParam()
    Meshflitband =  int(math.pow(2, round(math.log2(Meshflitband))))
    Htreeflitband =  int(math.pow(2, round(math.log2(Htreeflitband))))
    ADC_levels = float(2**Macroparameters["ADC_resolution"])    

    Set_Htreenums = [1, 4, 8, 16, 32, 64, 128, 256]
    Htreenums = Set_Htreenums
    Set_Htreesize = [[1, 1], [2, 2], [2, 4], [4, 4], [4, 8], [8, 8], [8, 16], [16, 16]]
    Set_Routernum = [0, 1, 3, 5, 11, 21, 43, 85]
    # Macronumbers = 64  # Set_macronums[i]
    Htreenums.append(Macronumbers)
    macronums = sorted(Htreenums)
    Macronums = macronums.index(Macronumbers)
    Macronumbers = Set_Htreenums[Macronums]
    RouternumperTile = Set_Routernum[Macronums]
    Tile = Set_Htreesize[Macronums]
    
    ### Update Params Here!!!(TODO: The order change too much. Need to be validated.)
    updateParam('SpecParam','Subarray',Subarray)
    updateParam('SpecParam','Tile',Tile)
    updateParam('SpecParam', 'buswidthTile', buswidthTile)
    updateParam('SpecParam', 'buffersizeTile', buffersizeTile)
    updateParam('SpecParam', 'ColumnMUX', ColumnMUX)
    updateParam('SpecParam', 'MeshNoC_flitband', Meshflitband)
    updateParam('SpecParam', 'HtreeNoC_flitband', Htreeflitband)
    
    curDir = os.getcwd()
    os.chdir("./refactor/build")
    os.system("./main --PPA_cost " + "user" + " " + "1") # any name/tid is ok
    os.chdir(curDir)

    energy = 0
    latency = 0
    area = 0
    if Subarray[0] < 512: # faster placing
        [total_energy, total_latency, total_area] = mapping(Spec_User_name,Spec_weight_name,'spec')
    else: # more accurate placing
        [total_energy, total_latency, total_area] = mapping(Spec_User_name,Spec_weight_name,'1')
    energy += total_energy
    latency += total_latency
    area += total_area
    if optparam['specification_optimized'] == True:
        perf_root_path = "../generate_data/" + Spec_User_name + "/performance_out"
        perf_path = perf_root_path + "/performance_out" + str(Spec_tid) + ".txt"
        with open(perf_path,"a+") as f:
            f.write("\n")
            f.write(f"total energy (mJ): \t{energy * 1000}\t total latency (ms): \t{latency * 1000}\t total area (mm2): \t{area}\n")
            f.write("\n")

    output = DefinedPerformance(energy, latency, area)
    return output


def Specification_optimizer(user_name,weight_name,tid):
    global Spec_User_name
    global Spec_weight_name
    global Spec_tid
    Spec_User_name = user_name
    Spec_weight_name = weight_name
    Spec_tid = tid

    start = time.perf_counter()
    verbose = 2
    optparam = OptParam()

    perf_root_path = "../generate_data/" + user_name + "/performance_out"
    perf_path = perf_root_path + "/performance_out" + str(tid) + ".txt"
    if optparam['specification_optimized'] == True:
        Trainfactors = trainParam()
        if Trainfactors["usermodel"] == True:
            path = "userdefined_cifar10_weight"
        if Trainfactors["defaultmodel"] == True:
            if Trainfactors["vgg11"]:
                path = "/vgg11_cifar10_weight"
            elif Trainfactors["vgg13"]:
                path = "/vgg13_cifar10_weight"
            elif Trainfactors["vgg16"]:
                path = "/vgg16_cifar10_weight"
            elif Trainfactors["vgg19"]:
                path = "/vgg19_cifar10_weight"
            elif Trainfactors["resnet18"]:
                path = "/resnet18_cifar10_weight"
            elif Trainfactors["resnet34"]:
                path = "/resnet34_cifar10_weight"
            elif Trainfactors["resnet50"]:
                path = "/resnet50_cifar10_weight"
            elif Trainfactors["resnet101"]:
                path = "/resnet101_cifar10_weight"
            elif Trainfactors["resnet152"]:
                path = "/resnet152_cifar10_weight"
            elif Trainfactors["mobilenet"]:
                path = "/mobilenet_cifar10_weight"
            elif Trainfactors["stgcn"]:
                path = "/stgcn_METRLA_weight"
        path = "Weight/" + user_name + weight_name + path
        # print(f'[INFO](Mapping_optimizer.py) path = {path}')
        resume = os.path.join(app_path(), path, "checkpoint.tar")
        if not os.path.isfile(resume):
            with open(perf_path,"w+") as f:
                f.write("No pretrained weights available. Please execute Retraining module first.")
                f.write("\n")
            return 0
        
        with open(perf_path,"w+") as f:
            f.write("=========================================================================================================================\n\n")
            f.write("################################Optimizing key specification for target DNN model######################################\n\n")
            f.write("=========================================================================================================================\n\n")
        # file = open(os.path.join(app_path(),"performance_out"+tid+".txt"), "w+")
        # sys.stdout = file

        # print(
        #     "=========================================================================================================================", flush=True)
        # print("################################Optimizing key specification for target DNN model######################################", flush=True)
        # print(
        #     "=========================================================================================================================", flush=True)
    verbose = 2
    Specboundaries = SpecboundParam()
    pbounds = {'Subarray': (Specboundaries['minSubarray'], Specboundaries['maxSubarray']),
               'Macronumbers': (Specboundaries['minMacronumbers'], Specboundaries['maxMacronumbers']),
               'buswidthTile': (Specboundaries['minbuswidthTile'], Specboundaries['maxbuswidthTile']),
               'buffersizeTile': (Specboundaries['minbuffersizeTile'], Specboundaries['maxbuffersizeTile']),
               'ColumnMUX': (Specboundaries['minColumnMUX'], Specboundaries['maxColumnMUX']),
               'Meshflitband': (Specboundaries['minmeshflitband'], Specboundaries['maxmeshflitband']),
               'Htreeflitband': (Specboundaries['minhtreeflitband'], Specboundaries['maxhtreeflitband'])}

    optimizer = BayesianOptimization(f=specification, pbounds=pbounds, random_state=1,verbose=verbose,
                                     path = perf_path)
    if optparam['circuit_optimized'] == True:
        optimizer.maximize(init_points=optparam['init_points']/10, n_iter=optparam['search_iters'])

    optimizer.maximize(init_points=optparam['init_points'], n_iter=optparam['search_iters'])

    arrayrow = int(math.pow(2, round(math.log2(optimizer.max['params']['Subarray']))))
    arraycol = int(arrayrow)
    Subarray = [arrayrow, arraycol]
    buffersizeTile = int(math.pow(2, round(math.log2(optimizer.max['params']['buffersizeTile']))))
    buswidthTile = int(math.pow(2, round(math.log2(optimizer.max['params']['buswidthTile']))))
    ColumnMUX = int(math.pow(2, round(math.log2(optimizer.max['params']['ColumnMUX']))))
    Set_Htreenums = [1, 4, 8, 16, 32, 64, 128, 256]
    Set_Htreesize = [[1, 1], [2, 2], [2, 4], [4, 4], [4, 8], [8, 8], [8, 16], [16, 16]]
    Set_Htreenums.append(optimizer.max['params']['Macronumbers'])
    macronums = sorted(Set_Htreenums)
    Macronums = macronums.index(optimizer.max['params']['Macronumbers'])
    Macronumbers = Set_Htreenums[Macronums]
    Tile = Set_Htreesize[Macronums]
    meshflitband = int(math.pow(2,round(math.log2(optimizer.max['params']['Meshflitband']))))
    htreeflitband = int(math.pow(2,round(math.log2(optimizer.max['params']['Htreeflitband']))))


    if optparam['specification_optimized'] == True:
        with open(perf_path,"a+") as f:
            f.write("IMC specifications are shown as follows\n\n")
            f.write(f"Subarray size: {Subarray}\n")
            f.write(f"Tile size: {Tile}\n")
            f.write(f"ADC numbers per Macro: {ColumnMUX}\n")
            f.write(f"Buswidth of Tile buffer (B): {buswidthTile}\n")
            f.write(f"Buffer size per Tile (KB): {buffersizeTile}\n")
            f.write(f"the sysytem performance: {optimizer.max['target']}\n\n")
        # print("Htree router flit bandwidth (B):", htreeflitband)
        # print("Mesh router flit bandwidth (B)", meshflitband)
        # file.close()
        # sys.stdout = temp



    end = time.perf_counter()


    return optimizer.max['target']

