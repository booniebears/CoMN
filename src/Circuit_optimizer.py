##################################################################################################
############          Scenario 3: Guading neuron and synapse for DNN model   ########################
##################################################################################################
# from GeneralizedPattern import *
from Parameters import *
from Specification_optimizer import *
import time
import sys
from models import *
from frozen_dir import app_path
# def NeruonSynapse(Network):
start = time.perf_counter()
temp = sys.stdout

global iter
global User_name
global Weight_name
iter = 0


def circuit(ADC_power, ADC_fre, ADC_area):
    global iter
    global User_name
    global Weight_name
    iter = iter + 1
    Macroparameters = MacroParam()
    Specparameters = SpecParam()
    optparam = OptParam()
    user_performance = float(optparam['target'])

    ADC_area = ADC_area * 1e-6
    updateParam('MacroParam',"ADC_power",ADC_power)
    updateParam('MacroParam', "ADC_fre", ADC_fre)
    updateParam('MacroParam', "ADC_area", ADC_area)

    curDir = os.getcwd()
    os.chdir("./refactor/build")
    os.system("./main --PPA_cost " + str(User_name) + " " + str(1)) # any name/tid is ok
    os.chdir(curDir)
    # PPA_cost(Macroparameters['MaxConductance'], Macroparameters['MinConductance'],
    #          Macroparameters['ADC_power'],
    #          Macroparameters['ADC_fre'], Macroparameters['ADC_area'],
    #          Specparameters['Subarray'], Specparameters['buswidthTile'],
    #          Specparameters['buffersizeTile'], Specparameters['ColumnMUX'],
    #          ADC_levels=2 ** Macroparameters['ADC_resolution'])
    if iter % 10 != 0:
        updateParam('OptParam','only_peupdate',True)
        [energy, latency, area] = mapping(User_name,Weight_name,'spec')
        output = DefinedPerformance(energy, latency, area)
    else:
        updateParam('OptParam', 'only_peupdate', False)
        output = Specification_optimizer(User_name,Weight_name,'1')
    min_out = abs(1/output - user_performance)
    return 1/min_out

def Circuit_optimizer(user_name,weight_name,tid):
    global User_name
    global Weight_name
    User_name = user_name
    Weight_name = weight_name

    verbose = 2
    # file = open(os.path.join(app_path(), "circuit_out"+tid+".txt"), "w+")
    # sys.stdout = file
    cir_root_path = "../generate_data/" + User_name + "/circuit_out"
    cir_path = cir_root_path + "/circuit_out" + str(tid) + ".txt"
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
        with open(cir_path,"w+") as f:
            f.write("No pretrained weights available. Please execute Retraining module first.")
            f.write("\n")
        return
    
    with open(cir_path,"w+") as f:
        f.write("====================================================================================================================\n\n")
        f.write("######################################Guiding neuron circuit or synapse device######################################\n\n")
        f.write("====================================================================================================================\n\n")
    
    # print(
    #     "====================================================================================================================\n",
    #     flush=True)
    # print(
    #         "######################################Guiding neuron circuit or synapse device######################################\n", flush=True)
    # print(
    #         "====================================================================================================================\n", flush=True)
    NeuroSynap = NeuronsynpaseParam()
    optparam = OptParam()
    Macroparameters = MacroParam()
    Specparameters = SpecParam()

    pbounds = {'ADC_power': (NeuroSynap['upboundADCpower'], NeuroSynap['downboundADCpower']),
               'ADC_fre': (NeuroSynap['upboundADCSR'], NeuroSynap['downboundADCSR']),
               'ADC_area': (NeuroSynap['upboundADCarea'], NeuroSynap['downboundADCarea'])}

    optimizer = BayesianOptimization(f=circuit, pbounds=pbounds, random_state=1, verbose=verbose,
                                     path=cir_path)

    optimizer.maximize(init_points=optparam['init_points'], n_iter=optparam['search_iters'])
    updateParam('MacroParam', "ADC_power", optimizer.max['params']['ADC_power'])
    updateParam('MacroParam', "ADC_fre", optimizer.max['params']['ADC_fre'])
    updateParam('MacroParam', "ADC_area", optimizer.max['params']['ADC_area'])
    
    curDir = os.getcwd()
    os.chdir("./refactor/build")
    os.system("./main --PPA_cost " + "user" + " " + str(1)) # any name/tid is ok
    os.chdir(curDir)
    # PPA_cost(Macroparameters['MaxConductance'], Macroparameters['MinConductance'],
    #          optimizer.max['params']['ADC_power'],
    #          optimizer.max['params']['ADC_fre'], optimizer.max['params']['ADC_area'],
    #          Specparameters['Subarray'], Specparameters['buswidthTile'],
    #          Specparameters['buffersizeTile'], Specparameters['ColumnMUX'],ADC_levels=2**Macroparameters['ADC_resolution'])
    Macro_performance = loadPerfParam("MacroPerf")
    # Macro_performance = np.load(os.path.join(app_path(), 'Performance/Macro.npy'), allow_pickle=True).item()
    
    with open(cir_path,"w+") as f:
        f.write("====================================================================================================================\n\n")
        f.write(f"ADC power (mW): {Macroparameters['ADC_power']*1000}\n")
        f.write(f"ADC sampling rate (GHz): {Macroparameters['ADC_fre']/1e9}\n")
        f.write(f"ADC active area (mm2): {Macroparameters['ADC_area']}\n")
        f.write(f"Macro computing energy (mJ): {Macro_performance['energy']*1000}\n")
        f.write(f"Macro computing latency (ms): {Macro_performance['latency']*1000}\n")
        f.write(f"Macro computing area (mm2): {Macro_performance['area']}\n")
        f.write(f"the sysytem performance: {optimizer.max['target']}\n")
        

    # file.close()
    # sys.stdout = temp
    updateParam('OptParam', 'only_peupdate', False)


