import os
import torch
from models.densenet import *
from models.googlenet import *
from models.vgg import *
from models.mobilenetv2 import *
from models.mobilenet import *
from models.resnet import *
from models.resnext import *
from models.xception import *
from models.inceptionv3 import *
from models.squeezenet import *
from models.stgcn import STGCN
from models.LeNet5 import Lenet5
from Network import *
from Parameters import *
from retrain_modules import *
from utils import *
from frozen_dir import app_path

def mapping(user_name, weight_name, tid):
    Trainfactors = trainParam()
    mappingOut_path = "../generate_data/" + user_name + "/mapping_out/mapping_out" + tid + ".txt"
    with open(os.path.join(app_path(), "Parameters/layer.txt"),"w") as f:
        f.write("0\n")
        f.write("0\n")
    if Trainfactors["usermodel"] == True:
        net = Net()
        path = "userdefined_cifar10_weight"
    if Trainfactors["defaultmodel"] == True:
        if Trainfactors["vgg11"]:
            net = vgg11()
            path = "/vgg11_cifar10_weight"
        elif Trainfactors["vgg13"]:
            net = vgg13()
            path = "/vgg13_cifar10_weight"
        elif Trainfactors["vgg16"]:
            net = vgg16()
            path = "/vgg16_cifar10_weight"
        elif Trainfactors["vgg19"]:
            net = vgg19()
            path = "/vgg19_cifar10_weight"
        elif Trainfactors["resnet18"]:
            net = resnet18()
            path = "/resnet18_cifar10_weight"
        elif Trainfactors["resnet34"]:
            net = resnet34()
            path = "/resnet34_cifar10_weight"
        elif Trainfactors["resnet50"]:
            net = resnet50()
            path = "/resnet50_cifar10_weight"
        elif Trainfactors["resnet101"]:
            net = resnet101()
            path = "/resnet101_cifar10_weight"
        elif Trainfactors["resnet152"]:
            net = resnet152()
            path = "/resnet152_cifar10_weight"
        elif Trainfactors["mobilenet"]:
            net = mobilenet()
            path = "/mobilenet_cifar10_weight"
        elif Trainfactors["lenet"]:
            net = Lenet5()
            path = "/lenet_mnist_weight"
        elif Trainfactors["stgcn"]:
            A, X, means, stds = load_metr_la_data()
            A_wave = get_normalized_adj(A)
            A_wave = torch.from_numpy(A_wave)
            # use past 12 data info to predict the future 3 data info
            num_timesteps_input = 12
            num_timesteps_output = 3
            split_line1 = int(X.shape[2] * 0.8)
            train_original_data = X[:, :, :split_line1]
            training_input, training_target = generate_dataset(train_original_data,
                                                        num_timesteps_input=num_timesteps_input,
                                                        num_timesteps_output=num_timesteps_output)
            net = STGCN(A_wave.shape[0], training_input.shape[3], num_timesteps_input,
                        num_timesteps_output)
            path = "/stgcn_METRLA_weight"
        
    model = net
    path = "Weight/" + user_name + weight_name + path
    resume = os.path.join(app_path(), path, "checkpoint.tar")
    # load checkpoint
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    else:
        with open(mappingOut_path, "w+", encoding="utf-8") as f:
            f.write("No pretrained weights available. Please execute Retraining module first.")
            f.write("\n")
        return 0,0,0
    optparam = OptParam()

    if Trainfactors["lenet"]:
        input = torch.load(os.path.join(app_path(), "Images/image_mnist.pth"))
    else:
        input = torch.load(os.path.join(app_path(), "Images/image.pth"))
    # print(f"[INFO](Mapping_optimizer.py) input.shape = {input.shape}") # torch.Size([1, 3, 32, 32])

    # *****************************************************************************************************
    if (
        optparam["specification_optimized"] == False
        and optparam["circuit_optimized"] == False
    ):
        # Evaluate unit PPA costs. Including Subarray, SFU, Buffer and Router.
        curDir = os.getcwd()
        os.chdir("./refactor/build")
        os.system("./main --PPA_cost " + str(user_name) + " " + str(tid))
        os.chdir(curDir)

        with open(mappingOut_path,"w") as f:
            f.write("#############################################################################################################\n\n")
            f.write("#######################    mapping CNN models to nvCIM chip and evaluating energy, latency and area    #######################\n\n")
            f.write("#############################################################################################################\n\n")
            
    with torch.no_grad():
        updateParam("OptParam", "evaluate_mode", True)
        updateParam("OptParam", "prepare_mode", True)

        os.system("rm " + os.path.join(app_path(), "Parameters/prepare.txt"))
        os.system("rm " + os.path.join(app_path(), "Parameters/placing.txt"))
        os.system("rm " + os.path.join(app_path(), "Parameters/layerconnect.txt"))
        if Trainfactors["stgcn"]:
            A_wave = get_normalized_adj(A)
            A_wave = torch.from_numpy(A_wave)
            size = (1, 207, 12, 2)
            tensor = torch.randn(size)
            output = model(A_wave, tensor)
        else:
            output = model(input)
        # To calculate duplication of each layer when applying different pipeline methods
        # Duplication nums are stored in Parameters/duplication.txt.
        
        command_str = "./main --pipeline_optimized " + str(user_name) + " " + str(tid)
        print(f"command : {command_str}")
        curDir = os.getcwd()
        os.chdir("./refactor/build")
        os.system("./main --pipeline_optimized " + str(user_name) + " " + str(tid))
        os.chdir(curDir)
        print("Pipeline finished!!!")
        # generate Performance/performance.txt when prepare_mode == False;
        updateParam("OptParam", "prepare_mode", False)
        os.system("rm " + os.path.join(app_path(), "Parameters/mapping.txt"))
        os.system("rm " + os.path.join(app_path(), "Performance/performance.txt"))
        os.system("rm " + os.path.join(app_path(), "Performance/SFU_performance.txt"))
        os.system("rm " + os.path.join(app_path(), "Parameters/traffic.txt"))
        os.system(
            "mv "
            + os.path.join(app_path(), "Parameters/layerconnect.txt")
            + " "
            + os.path.join(app_path(), "Parameters/meshconnect.txt")
        )
        with open(os.path.join(app_path(), "Parameters/tileNum.txt"),"w") as f:
            f.write(str(0))
        if Trainfactors["stgcn"]:
            A_wave = get_normalized_adj(A)
            A_wave = torch.from_numpy(A_wave)
            size = (1, 207, 12, 2)
            tensor = torch.randn(size)
            output = model(A_wave, tensor)
        else:
            output = model(input)
        curDir = os.getcwd()
        os.chdir("./refactor/build")
        os.system("./main --Mesh_operation " + str(user_name) + " " + str(tid))
        os.chdir(curDir)
        
        energy = ""
        latency = ""
        area = ""
        with open(os.path.join(app_path(), "Performance/performance.txt"),"r") as f:
            for line in f:
                blocks = line.split()
                if len(blocks) == 9 and blocks[0] == 'total': # The line we desire
                    energy = float(blocks[2])
                    latency = float(blocks[5])
                    area = float(blocks[8])
        
    return energy, latency, area

def mapping_optimizer(user_name, weight_name, tid):
    updateParam("OptParam", "mapping_finish", False)

    updateParam("OptParam", "specification_optimized", False)
    updateParam("OptParam", "circuit_optimized", False)
    # Trainfactors = trainParam()
    Macroparameters = MacroParam()
    Specparameters = SpecParam()
    mappingOut_path = "../generate_data/" + user_name + "/mapping_out/mapping_out" + tid + ".txt"
    do_mapping = True
    # print(f"Path Exists for mapping: {os.path.exists(mappingOut_path)}")
    if Macroparameters["NVM_states"] <= 1 or Macroparameters["NVM_states"] > 8:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("The conductance states per NVM is too large")
            f.write("\n")
        do_mapping = False
    if Specparameters["Subarray"][0] > 1024 or Specparameters["Subarray"][1] > 1024:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write(
                "The subarray size is too large, 128*128 subarray size is recommended"
            )
            f.write("\n")
        do_mapping = False
    if Specparameters["Subarray"][0] < 64 or Specparameters["Subarray"][1] < 64:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write(
                "The subarray size is too small, 128*128 subarray size is recommended"
            )
            f.write("\n")
        do_mapping = False
    if Specparameters["ColumnMUX"] < 1:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("Not less than one ADC per Macro are required")
            f.write("\n")
        do_mapping = False
    if Specparameters["Tile"][0] > 16 or Specparameters["Tile"][1] > 16:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("There are too many macros per Tile")
            f.write("\n")
        do_mapping = False
    if Specparameters["Tile"][0] < 2 or Specparameters["Tile"][1] < 2:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("Not less than 4 macros per Tile are required")
            f.write("\n")
        do_mapping = False
    if Specparameters["buffersizeTile"] < 2:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("Not less than 2KB buffer are required per Tile")
            f.write("\n")
        do_mapping = False
    if Specparameters["buswidthTile"] > 128 or Specparameters["buswidthTile"] < 8:
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("Appropriate buffer bandwidth are required")
            f.write("\n")
        do_mapping = False
    if (
        Specparameters["MeshNoC_flitband"] > 128
        or Specparameters["MeshNoC_flitband"] < 8
    ):
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("Appropriate Mesh NoC bandwidth are required")
            f.write("\n")
        do_mapping = False
    # print(f'Specparameters["HtreeNoC_flitband"] = {Specparameters["HtreeNoC_flitband"]}')
    if (
        Specparameters["HtreeNoC_flitband"] > 128
        or Specparameters["HtreeNoC_flitband"] < 8
    ):
        with open(mappingOut_path, "a+", encoding="utf-8") as f:
            f.write("Appropriate Htree NoC bandwidth are required")
            f.write("\n")
        do_mapping = False
        
    if do_mapping:
        mapping(user_name, weight_name, tid)
    updateParam("OptParam", "mapping_finish", True)
    updateParam("OptParam", "specification_optimized", True)


if __name__ == "__main__":
    mapping_optimizer("tcad", "", "1")
