/**
 * @file Perf_Evaluator.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <ctime>
#include <chrono>

#include "PE.h"
#include "Perf_Evaluator.h"
#include "defines.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

namespace Refactor {

void PPA_cost() {
  // 1: Obtain config params from json files;
  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  ifstream f_macro(PATH_MACRO_PARAM);
  json macroParam = json::parse(f_macro);
  double MaxConductance = macroParam["MaxConductance"],
         MinConductance = macroParam["MinConductance"],
         ADC_power = macroParam["ADC_power"], ADC_fre = macroParam["ADC_fre"],
         ADC_area = macroParam["ADC_area"];
  vector<int> Subarray(2);
  Subarray[0] = specParam["Subarray"][0];
  Subarray[1] = specParam["Subarray"][1];
  // ADCNum: num of ADCs in a macro;
  int buswidthTile = specParam["buswidthTile"],
      buffersizeTile = specParam["buffersizeTile"],
      ADCNum = specParam["ColumnMUX"], ADCLevel = macroParam["ADC_resolution"];
  ADCLevel = pow(2, ADCLevel);

  // 2: Calling the performance evaluator of different modules
  ifstream f_opt(PATH_OPT_PARAM);
  json optParam = json::parse(f_opt);
  ifstream f_tech(PATH_TECH_PARAM);
  json tech = json::parse(f_tech);
  int techNode = tech["featuresize"];
  if (!optParam["only_peupdate"]) {
    Buffer_Perf(buffersizeTile * 1024, buswidthTile, techNode);
    // For NoC_mesh
    Orion_Perf(specParam["MeshNoC_flitband"], 5, 5, 4, 1e9, techNode, true);
    // For NoC_Htree
    Orion_Perf(specParam["HtreeNoC_flitband"], 5, 5, 4, 1e9, techNode,
               false);
  }
  PEInfo info{Subarray, MaxConductance, MinConductance, ADCNum,
              ADCLevel, ADC_power,      ADC_fre,        ADC_area};
  PE_core_energy(info, techNode);
}

void Buffer_Perf(int bufferSize, int buswidth, int featureSize) {
  cout << "Into Buffer_Perf!!!" << endl;

  // modify the cache.cfg
  ifstream f_cfg(PATH_CACTI_CFG);
  ofstream f_new(PATH_CACTI_CFG_BAK);
  string line;
  while (getline(f_cfg, line)) {
    if (line.find("-size (bytes) 131072") != string::npos) {
      line = line.replace(line.find("131072"), 6, to_string(bufferSize));
    } else if (line.find("-output/input bus width 512") != string::npos) {
      line = line.replace(line.find("512"), 3, to_string(buswidth));
    }
    f_new << line << endl;
  }

  // cout << "Conducting Before cacti!" << endl;
  auto curDir = get_current_dir_name();
  chdir("../../../cacti-master");
  system("./cacti -infile cache_bak.cfg > output.txt");
  chdir(curDir);
  // cout << "Conducting After cacti!" << endl;

  // Results stored in bufferInfo.txt
  ifstream f_buffer(PATH_BUFFER);
  json bufferPerf;
  while (getline(f_buffer, line)) {
    istringstream iss(line);
    string key, value;
    getline(iss, key, ':');
    getline(iss, value);
    key.erase(key.find_last_not_of(" ") + 1);
    value.erase(0, key.find_first_not_of(" "));
    // TODO: Perf calculation to be further discussed
    if (key == "latency") {
      bufferPerf["latency"] =
          (stod(value) / buswidth) * featureSize / 22 * featureSize / 22 * 1e-9;
    } else if (key == "read_energy") {
      bufferPerf["read_energy"] =
          (stod(value) / buswidth) * featureSize / 22 * 1e-9;
    } else if (key == "write_energy") {
      bufferPerf["write_energy"] =
          (stod(value) / buswidth) * featureSize / 22 * 1e-9;
    } else if (key == "area") {
      bufferPerf["area"] = stod(value) * featureSize / 22 * featureSize / 22;
    }
  }

  ofstream f_SPMPerf(PATH_SPM_PERF);
  f_SPMPerf << setw(2) << bufferPerf << endl;
  f_SPMPerf.close();
}

void Orion_Perf(int Fliter_size, int inPorts, int outPorts, int v_channels,
                double freq, int featureSize, bool isMesh) {
  ifstream f_cfg(PATH_ORION_CFG);
  ofstream f_new(PATH_ORION_NEW);
  string line;
  while (getline(f_cfg, line)) {
    if (line.find("#define PARM_flit_width		16") != string::npos) {
      line = line.replace(line.find("16"), 2, to_string(Fliter_size));
    } else if (line.find("#define PARM_in_port 		3") != string::npos) {
      line = line.replace(line.find("3"), 1, to_string(inPorts));
    } else if (line.find("#define PARM_out_port		3") != string::npos) {
      line = line.replace(line.find("3"), 1, to_string(outPorts));
    } else if (line.find("#define PARM_v_channel		4") !=
               string::npos) {
      line = line.replace(line.find("4"), 1, to_string(v_channels));
    }
    f_new << line << endl;
  }
  auto curDir = get_current_dir_name();
  chdir("../../../ORION3_0");
  system("make > makeInfo.txt");
  // system("mv *.o build/");
  system("./orion_router > output.txt");
  chdir(curDir);
  ifstream f_noc(PATH_ORION_OUT);

  double energy = 0, area = 0;
  getline(f_noc, line); // read useless info first
  while (getline(f_noc, line)) {
    istringstream iss(line);
    string key, value;
    getline(iss, key, ':');
    getline(iss, value);
    key.erase(key.find_last_not_of(" ") + 1);
    value.erase(0, key.find_first_not_of(" "));
    if (key == "Ptotal") {
      // TODO: why calculated this way ??
      energy = stod(value) / 8 / freq / (inPorts + outPorts) * featureSize /
               65 * 1e-3;
    } else if (key == "Atotal") {
      area = stod(value) / 1e6 * featureSize / 65 * featureSize / 65;
    }
  }
  if (isMesh) {
    json MeshPerf;
    MeshPerf["energy"] = energy;
    MeshPerf["area"] = area;
    MeshPerf["latency"] =
        1 / freq / v_channels * featureSize / 65 * featureSize / 65;
    ofstream of_mesh(PATH_MESH_PERF);
    of_mesh << setw(2) << MeshPerf << endl;
  } else {
    json HtreePerf;
    HtreePerf["energy"] = energy;
    HtreePerf["area"] = area;
    HtreePerf["latency"] =
        1 / freq / v_channels * featureSize / 65 * featureSize / 65;
    ofstream of_htree(PATH_HTREE_PERF);
    of_htree << setw(2) << HtreePerf << endl;
  }
}

} // namespace Refactor
