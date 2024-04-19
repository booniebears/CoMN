from Parameters import *
from Circuit_optimizer import *
from Specification_optimizer import *
import sys

temp = sys.stdout

def performance_optimizer(user_name,weight_name,tid):
    optparam = OptParam()
    Specparameters = SpecboundParam()
    if optparam['specification_optimized'] == True:
        perf_root_path = "../generate_data/" + user_name + "/performance_out"
        perf_path = perf_root_path + "/performance_out" + str(tid) + ".txt"
        if Specparameters['minSubarray'] < 32 or Specparameters['minSubarray'] > 512:
            with open(perf_path, 'w+', encoding="utf-8") as f:
                f.write('Minimum subarray size error ')
                f.write("\n")
            exit()
        if Specparameters['maxSubarray'] < 64 or Specparameters['maxSubarray'] > 1024:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum subarray size error')
                f.write("\n")
            exit()
        if Specparameters['minColumnMUX'] < 1 or Specparameters['minColumnMUX'] > 32:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum ADC numbers error')
                f.write("\n")
            exit()
        if Specparameters['maxColumnMUX'] < 1 or Specparameters['maxColumnMUX'] > 64:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum ADC numbers error')
                f.write("\n")
            exit()
        if Specparameters['minMacronumbers'] < 2 or Specparameters['minMacronumbers'] > 16:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum Macro numbers per Tile error')
                f.write("\n")
            exit()
        if Specparameters['maxMacronumbers'] < 2 or Specparameters['maxMacronumbers'] > 16:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum Macro numbers per Tile error')
                f.write("\n")
            exit()
        if Specparameters['minbuffersizeTile'] < 2 or Specparameters['minbuffersizeTile'] > 128:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum buffer size per Tile error')
                f.write("\n")
            exit()
        if Specparameters['maxbuffersizeTile'] < 2 or Specparameters['maxbuffersizeTile'] > 128:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum buffer size per Tile error')
                f.write("\n")
            exit()
        if Specparameters['minbuswidthTile'] > 128 or Specparameters['minbuswidthTile'] < 8:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Minimum buffer bandwidth error')
                f.write("\n")
            exit()
        if Specparameters['maxbuswidthTile'] > 128 or Specparameters['maxbuswidthTile'] < 8:
            with open(perf_path, 'a+', encoding="utf-8") as f:
                f.write('Maximum buffer bandwidth error')
                f.write("\n")
            exit()
        Specification_optimizer(user_name,weight_name,tid)


    if optparam['circuit_optimized'] == True:
        updateParam('MacroParam', "predefinedMacro", False)
        Circuit_optimizer(user_name,weight_name,tid)

def performance(user_name,weight_name,tid):
    updateParam('OptParam', 'guiding_finish', False)

    performance_optimizer(user_name,weight_name,tid)
    updateParam('OptParam', 'guiding_finish', True)

if __name__ == '__main__':
    performance("tcad","",'1')