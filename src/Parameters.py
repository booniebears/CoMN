import json
import os
from frozen_dir import app_path

def layer_connect(filename, prelayer, nextlayer, type, volumn,Dir="Parameters"):
    opt = OptParam(Dir=Dir)
    if opt["evaluate_mode"] == True:
        if len(volumn) == 4:
            volumn = volumn[0] * volumn[1] * volumn[2] * volumn[3]
        else:
            volumn = volumn[0] * volumn[1]
        if prelayer > 0:
            with open(
                os.path.join(app_path(), "Parameters/{}.txt").format(filename),
                "a+",
                encoding="utf-8",
            ) as f:
                f.write(
                    "prelayer: {prelayer}\t"
                    "nextlayer: {nextlayer}\t"
                    "type: {type}\t"
                    "volumn: {volumn}\t"
                    "prelayer_tile: {tiles1} {tiles2} \t"
                    "nextlayer_tile: {tiles3} {tiles4} \n".format(
                        prelayer=prelayer,
                        nextlayer=nextlayer,
                        type=type,
                        volumn=volumn,
                        tiles1=0,
                        tiles2=0,
                        tiles3=0,
                        tiles4=0,
                    )
                )


def loadParam(param,Dir="Parameters"):
    with open(
        os.path.join(app_path(), Dir,"{}.json").format(param),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.loads(f.read())
        return data

def loadPerfParam(param):
    with open(
        os.path.join(app_path(), "Performance/{}.json").format(param),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.loads(f.read())
        return data


def saveParam(param, udata):
    with open(
        os.path.join(app_path(), "Performance/{}.json").format(param),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(udata))


def updateParam(param, paramname, udata, Dir="Parameters"):
    dict_data = loadParam(param,Dir=Dir)
    dict_data[paramname] = udata
    with open(
        os.path.join(app_path(), Dir,"{}.json").format(param),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(dict_data))


def TechnodeParam():
    Tech = loadParam("TechnodeParam")
    return Tech


def MacroParam(Dir="Parameters"):
    Macroparameters = loadParam("MacroParam",Dir=Dir)

    return Macroparameters


# print(type(MacroParam()['Rsense']))
def NNParam():
    # [inchannels,outchannels,featuresize,featuresize,kernelsize,kernelsize,stride,padding,type)
    # Vgg11 = [[['Conv',1, 6, 28, 28, 3, 3, 1, 1], []],
    #          [['ConvRelu',1, 1, 1, 1, 1, 1, 1, 1], []],
    #          [['Maxpool',6, 6, 28, 28, 2, 2, 2, 0], []],
    #          [['Conv',6, 16, 14, 14, 3, 3, 1, 1], []],
    #          [['ConvRelu',1, 1, 1, 1, 1, 1, 1, 1], []],
    #          [['Maxpool',16, 16, 14, 14, 2, 2, 2, 0], []],
    #
    #          [['Linear',784, 120, 1, 1, 1, 1, 1, 1], []],
    #          [['LinearRelu',1, 1, 1, 1, 1, 1, 1, 1], []],
    #          [['Linear',120, 84, 1, 1, 1, 1, 1, 1], []],
    #          [['LinearRelu',1, 1, 1, 1, 1, 1, 1, 1], []],
    #          [['Linear',84, 10, 1, 1, 1, 1, 1, 1], []]]

    with open("../Parameters/NNParam.txt", "r", encoding="utf-8") as f:
        data = f.read()
    Vgg11 = eval(data)

    return Vgg11


def trainParam(Dir="Parameters"):
    Trainfactors = loadParam("trainParam",Dir=Dir)

    return Trainfactors


def SpecParam(Dir="Parameters"):
    Specification = loadParam("SpecParam",Dir=Dir)

    return Specification


def SpecboundParam():
    Specboundaries = loadParam("SpecboundParam")
    return Specboundaries


def NeuronsynpaseParam():
    NeurSynap = loadParam("NeuronsynpaseParam")

    return NeurSynap


def OptParam(Dir="Parameters"):
    Opt = loadParam("OptParam",Dir=Dir)
    return Opt


def DefinedPerformance(energy, latency, area):
    # latency unit is second, energy unit is mJ, area unit is mm2
    Opt = OptParam()
    if Opt["energy"] == True:
        optimizedperformance = energy
    if Opt["latency"] == True:
        optimizedperformance = latency
    if Opt["area"] == True:
        optimizedperformance = area
    if Opt["latency"] == True and Opt["energy"] == True:
        optimizedperformance = energy / latency
    if Opt["area"] == True and Opt["energy"] == True:
        optimizedperformance = energy * area
    if Opt["area"] == True and Opt["latency"] == True:
        optimizedperformance = latency * area
    if Opt["energy"] == True and Opt["area"] == True and Opt["latency"] == True:
        optimizedperformance = latency * area * energy

    return 1 / optimizedperformance
