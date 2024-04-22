import numpy as np
import os
import math
from Parameters import TechnodeParam
from frozen_dir import app_path
class Buffer_performance:


    def __init__(self, Buffer_size, buswidth):
        self.Buffer_size = Buffer_size*1024
        self.buswidth = buswidth


    def Buffer_energy(self):
        Tech = TechnodeParam()
        if self.Buffer_size <= 1024:
            print("Please set up larger buffer size")

        else:
            f_cfg = open(os.path.join(app_path(), 'cacti-master/cache.cfg'), 'r')
            # open new file
            f_new = open(os.path.join(app_path(), 'cacti-master/cache_bak.cfg'), 'w')
            # read old file circularly
            for line in f_cfg:
                if "-size (bytes) 131072" in line:
                    print(f'Buffer_size = {self.Buffer_size}')
                    line = line.replace('131072', str(self.Buffer_size))
                    # print(f'[INFO] (Buffer_module.py)')
                # if "-block size (bytes) 128" in line:
                #     line = line.replace('128', str(self.block_size/8))
                if "-output/input bus width 512" in line:
                    print(f'buswidth = {self.buswidth}')
                    line = line.replace('512', str(self.buswidth))
                # write old file to new file
                f_new.write(line)
            f_cfg.close()
            f_new.close()

            Buffer_Performance = dict()
            os.chdir(os.path.join(app_path(), 'cacti-master'))
            os.system("./cacti -infile cache_bak.cfg > output.txt")
            # os.system(os.path.join(app_path(), 'cacti-master/cacti')+' -infile cacti-master/cache_bak.cfg > '+ os.path.join(app_path(), 'cacti-master/output.txt'))

            f_output = open(os.path.join(app_path(), 'cacti-master/output.txt'), 'r')
            # TODO: read bufferInfo.txt later in C++; remove some code in cacti-master/io.cc.
            for line in f_output:
                if " Access time (ns): " in line:
                    a = line.split()
                    # print(f'Cacti latency = {float(np.array(a[3:4]))}')
                    Buffer_Performance['latency'] = float(np.array(a[3:4]))/self.buswidth
                    Buffer_Performance['latency'] *= Tech['featuresize']/22*Tech['featuresize']/22*1e-9
                if "Total dynamic read energy per access (nJ):" in line:
                    a = line.split()
                    # print(f'Cacti read_energy = {float(np.array(a[7:8]))}')
                    Buffer_Performance['read_energy'] = float(np.array(a[7:8]))/self.buswidth  #energy per bit
                    Buffer_Performance['read_energy'] *= Tech['featuresize']/22
                if "Total dynamic write energy per access (nJ):" in line:
                    a = line.split()
                    # print(f'Cacti write_energy = {float(np.array(a[7:8]))}')
                    Buffer_Performance['write_energy'] = float(np.array(a[7:8]))/ self.buswidth
                    Buffer_Performance['write_energy'] *= Tech['featuresize'] / 22
                if "Cache height x width (mm):" in line:
                    a = line.split()
                    # print(f'Cacti area = {float(np.array(a[5:6])) * float(np.array(a[7:8]))}')
                    Buffer_Performance['area'] = float(np.array(a[5:6])) * float(np.array(a[7:8]))
                    Buffer_Performance['area'] *= Tech['featuresize']/22*Tech['featuresize']/22
            f_output.close()
            
            # Cacti latency = 0.446677
            # Cacti read_energy = 0.0135347
            # Cacti write_energy = 0.0114108
            # Cacti area = 0.15796613705399998

        return Buffer_Performance