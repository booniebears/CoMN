import numpy as np
import os
import math
from Parameters import TechnodeParam
from frozen_dir import app_path

class NoC_Orion:
    def __init__(self, Fliter_size, inPorts,outPorts, v_channels,Freq):
        self.fliter_size = Fliter_size
        self.vchannels = v_channels
        self.inports = inPorts
        self.outports = outPorts
        self.freq = Freq
    def Orion(self):
        Tech = TechnodeParam()
        f_cfg = open(os.path.join(app_path(), 'ORION3_0/SIM_port_cfg.h'), 'r')
        f_new = open(os.path.join(app_path(), 'ORION3_0/SIM_port.h'), 'w')
        for line in f_cfg:
            if "#define PARM_flit_width		16" in line:
                # line = line.replace('16', '4')
                line = line.replace('16', str(self.fliter_size))
            if "#define PARM_in_port 		3" in line:
                line = line.replace('3', str(self.inports))
            if "#define PARM_out_port		3" in line:
                line = line.replace('3', str(self.outports))
            if "#define PARM_v_channel		4" in line:
                line = line.replace('3', str(self.vchannels))
            f_new.write(line)
        f_cfg.close()
        f_new.close()
        NoC_PowerArea = dict()
        # os.chdir('ORION3_0')
        # os.system('make clean')
        os.chdir(os.path.join(app_path(), 'ORION3_0'))
        os.system('make > makeInfo.txt')
        os.system('mv *.o build/')
        os.system('./orion_router > output.txt')
        # os.chdir('cd ' +app_path())
        f_output = open(os.path.join(app_path(), 'ORION3_0/output.txt'), 'r')
        # TODO: read ORION3_0/output.txt in C++;
        for line in f_output:
            if "Ptotal" in line:
                a = line.split()
                NoC_PowerArea['energy'] = float(np.array(a[1:2])) /8 /self.freq / (self.inports + self.outports)*Tech['featuresize']/65 * 1e6  # unit: nJ
            if "Atotal:" in line:
                a = line.split()
                NoC_PowerArea['area'] = float(np.array(a[1:2]))/1e6*Tech['featuresize']/65*Tech['featuresize']/65 #unit: mm^2
        f_output.close()
        NoC_PowerArea['latency'] = 1/self.freq/self.vchannels*Tech['featuresize']/65*Tech['featuresize']/65

        return NoC_PowerArea

