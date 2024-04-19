/**
 * @file PE.cpp
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

#include "PE.h"
#include "defines.h"

#include "CoMN_Sim/include/BitShifter.h"
#include "CoMN_Sim/include/MaxPool.h"
#include "CoMN_Sim/include/Sigmoid.h"
#include "CoMN_Sim/include/SynapticArray.h"
#include "CoMN_Sim/include/general/Constant.h"
#include "CoMN_Sim/include/general/Formula.h"
#include "CoMN_Sim/include/general/Param.h"
#include "CoMN_Sim/include/general/Technology.h"

using namespace CoMN;

namespace Refactor {

void PE_core_energy(PEInfo &info, int techNode) {
  ofstream f_macroPerf(PATH_MACRO_PERF);
  ofstream f_reluPerf(PATH_RELU_PERF);
  ofstream f_maxpoolPerf(PATH_MAXPOOL_PERF);
  ofstream f_sigmoidPerf(PATH_SIGMOID_PERF);
  json macroPerf, reluPerf, maxpoolPerf, sigmoidPerf;

  ifstream f_macro(PATH_MACRO_PARAM);
  ifstream f_opt(PATH_OPT_PARAM);
  json macroParam = json::parse(f_macro);
  json optParam = json::parse(f_opt);
  if (macroParam["predefinedMacro"]) {
    cout << "Getting perf for predefinedMacro!!" << endl;
    macroPerf["energy"] = macroParam["Macro_energy"];
    macroPerf["latency"] = macroParam["Macro_delay"];
    macroPerf["area"] = macroParam["Macro_area"];
    // TODO: why set the coefficients as such??
    reluPerf["energy"] = 1.411e-15 * techNode / 22;
    reluPerf["latency"] = 5e-9 * techNode / 22 * techNode / 22;
    reluPerf["area"] = 1.236e-6 * techNode / 22 * techNode / 22;
    maxpoolPerf["energy"] = 6.969e-15 * techNode / 22;
    maxpoolPerf["latency"] = 2.180e-10 * techNode / 22 * techNode / 22;
    maxpoolPerf["area"] = 9.888e-6 * techNode / 22 * techNode / 22;
    sigmoidPerf["energy"] = 1.001e-14 * techNode / 22;
    sigmoidPerf["latency"] = 2.01e-8 * techNode / 22 * techNode / 22;
    sigmoidPerf["area"] = 1.605e-5 * techNode / 22 * techNode / 22;
    f_macroPerf << setw(2) << macroPerf << endl;
    f_reluPerf << setw(2) << reluPerf << endl;
    f_maxpoolPerf << setw(2) << maxpoolPerf << endl;
    f_sigmoidPerf << setw(2) << sigmoidPerf << endl;
  } else {
    CoMNInfo CoMNinfo;

    CoMNinfo.technode = techNode;
    CoMNinfo.numRowSubArray = info.Subarray[0];
    CoMNinfo.numColSubArray = info.Subarray[1];
    CoMNinfo.levelOutput = info.ADCLevel;
    CoMNinfo.numColMuxed = info.Subarray[1] / info.ADCNum;
    CoMNinfo.featuresize = techNode * 1e-9;
    CoMNinfo.readPulseWidth = macroParam["readPulseWidth"];
    CoMNinfo.readVoltage = macroParam["readVoltage"];
    CoMNinfo.resistanceOn = 1.0 / info.MaxConductance;
    CoMNinfo.resistanceOff = 1.0 / info.MinConductance;
    CoMNinfo.ADC_power = info.ADC_power;
    CoMNinfo.ADC_delay = 1.0 / info.ADC_fre;
    CoMNinfo.ADC_area = info.ADC_area;
    CoMNinfo.user_defined = (optParam["circuit_optimized"] == true);

    CoMN_interface(CoMNinfo);
  }
}

void CoMN_interface(CoMNInfo &info) {
  Param *param = new Param();
  // 1: Transfer values from CoMNInfo{} to Param{}
  param->technode = info.technode;
  param->numRowSubArray = info.numRowSubArray;
  param->numColSubArray = info.numColSubArray;
  param->levelOutput = info.levelOutput;
  param->numColMuxed = info.numColMuxed;
  param->featureSize = info.featuresize;
  param->readPulseWidth = info.readPulseWidth;
  param->readVoltage = info.readVoltage;
  param->resistanceOn = info.resistanceOn;
  param->resistanceOff = info.resistanceOff;
  param->ADC_power = info.ADC_power;
  param->ADC_delay = info.ADC_delay;
  param->ADC_area = info.ADC_area;
  param->user_defined = info.user_defined;

  // 2: Executing the original CoMN_Sim code
  Technology *tech = new Technology();
  tech->Initialize(param->technode, param->deviceRoadmap,
                   param->transistorType);
  tech->featureSize = param->featureSize;
  SynapticArray *synapticArray = new SynapticArray(param, tech);
  BitShifter *relu = new BitShifter(param, tech);
  MaxPool *maxpool = new MaxPool(param, tech);
  Sigmoid *sigmoid = new Sigmoid(param, tech);

  // Configure NN conditions
  synapticArray->conventionalSequential = param->conventionalSequential;
  synapticArray->conventionalParallel = param->conventionalParallel;
  synapticArray->BNNsequentialMode = param->BNNsequentialMode;
  synapticArray->BNNparallelMode = param->BNNparallelMode;
  synapticArray->XNORsequentialMode = param->XNORsequentialMode;
  synapticArray->XNORparallelMode = param->XNORparallelMode;

  // inputVector.shape = (param->numRowSubArray,numInVector), numInVector =
  // param->numBitInput * (kernel moves on input map (no padding))
  // Only one Subarray is considered, so we can initialize as below:
  int numInVector = 1;
  vector<vector<double>> inputVector(param->numRowSubArray,
                                     vector<double>(numInVector, 1));
  vector<vector<double>> subArrayMemory(
      param->numRowSubArray,
      vector<double>(param->numColSubArray, param->maxConductance));
  // for (int i = 0; i < numInVector; i++) {
  double activityRowRead = 0;
  vector<double> input;
  input = GetInputVector(inputVector, 0, &activityRowRead);
  synapticArray->activityRowRead = activityRowRead;
  // cout << "synapticArray->activityRowRead = " <<
  // synapticArray->activityRowRead
  //      << endl;
  // }

  // Configure Array parameters
  if (param->XNORparallelMode || param->XNORsequentialMode) {
    param->numRowPerSynapse = 2;
  } else {
    param->numRowPerSynapse = 1;
  }
  if (param->BNNparallelMode) {
    param->numColPerSynapse = 2;
  } else if (param->XNORparallelMode || param->XNORsequentialMode ||
             param->BNNsequentialMode) {
    param->numColPerSynapse = 1;
  } else {
    param->numColPerSynapse =
        ceil((double)param->synapseBit / (double)param->cellBit); // 目前为1
  }
  // cout << "param->numColPerSynapse = " << param->numColPerSynapse << endl;
  // cout << "unitLengthWireResistance = " << param->unitLengthWireResistance
  //      << endl;

  // Configure Array Parameters
  synapticArray->numOfCellsPerSynapse = param->numColPerSynapse;
  synapticArray->numOfReadPulses = param->numBitInput;
  synapticArray->spikingMode = NONSPIKING;
  synapticArray->numOfColumnsMux = param->numColMuxed;
  synapticArray->isSarADC = param->SARADC;
  synapticArray->isCSA = param->currentMode;
  synapticArray->avgWeightBit = param->cellBit;
  synapticArray->widthAccessCMOS = param->widthAccessCMOS;
  synapticArray->resistanceAvg =
      (param->resistanceOn + param->resistanceOff) / 2;

  synapticArray->Initialize(param->numColSubArray, param->numRowSubArray,
                            param->unitLengthWireResistance);
  synapticArray->CalculateArea(); // TODO: check area/latency/power cal.

  // For each column in synaptic array, get a equivalent resistance.
  vector<double> columnResistance =
      GetColumnResistance(input, subArrayMemory, param->parallelRead,
                          synapticArray->resCellAccess, param);

  synapticArray->CalculatePower(columnResistance);
  synapticArray->CalculateLatency(INF_RAMP, columnResistance);

  // cout << "synapticArray performance: " << endl;
  // cout << "synapticArray Energy (nJ): "
  //      << synapticArray->readDynamicEnergy * 1e9 << endl;
  // cout << "synapticArray Latency (s): " << synapticArray->readLatency <<
  // endl; cout << "synapticArray Area (mm^2): " << synapticArray->usedArea *
  // 1e6
  //      << endl;

  relu->Initialize(param->numBitInput, 1);
  relu->CalculateArea(0, synapticArray->width, NONE);
  relu->CalculateLatency(1); // numRead = 1
  relu->CalculatePower(1);
  // cout << "relu performance: " << endl;
  // cout << "relu Energy (nJ): " << relu->readDynamicEnergy * 1e9 << endl;
  // cout << "relu Latency (s): " << relu->readLatency << endl;
  // cout << "relu Area (mm^2): " << relu->area * 1e6 << endl;

  maxpool->Initialize(param->numBitInput, MAXPOOL_WINDOW, 1);
  maxpool->CalculateArea(synapticArray->width);
  maxpool->CalculateLatency(INF_RAMP, 0, 1); // numRead = 1
  maxpool->CalculatePower(1);

  // TODO: not sure how to calculate the num of entries in Sigmoid lookup table.
  int numEntries = ceil(log2(param->levelOutput)) + param->numBitInput +
                   param->numColPerSynapse + 1;
  sigmoid->Initialize(param->numBitInput, numEntries, 1);
  sigmoid->CalculateArea(synapticArray->width, synapticArray->height, NONE);
  sigmoid->CalculateLatency(1);
  sigmoid->CalculatePower(1);

  ofstream f_macroPerf(PATH_MACRO_PERF);
  ofstream f_reluPerf(PATH_RELU_PERF);
  ofstream f_maxpoolPerf(PATH_MAXPOOL_PERF);
  ofstream f_sigmoidPerf(PATH_SIGMOID_PERF);
  json macroPerf, reluPerf, maxpoolPerf, sigmoidPerf;
  macroPerf["area"] = synapticArray->usedArea * 1e6;      // mm^2
  macroPerf["latency"] = synapticArray->readLatency; // s
  macroPerf["energy"] = synapticArray->readDynamicEnergy; // J
  reluPerf["area"] = relu->area * 1e6;
  reluPerf["latency"] = relu->readLatency;
  reluPerf["energy"] = relu->readDynamicEnergy;
  maxpoolPerf["area"] = maxpool->area * 1e6;
  maxpoolPerf["latency"] = maxpool->readLatency;
  maxpoolPerf["energy"] = maxpool->readDynamicEnergy;
  sigmoidPerf["area"] = sigmoid->area * 1e6;
  sigmoidPerf["latency"] = sigmoid->readLatency;
  sigmoidPerf["energy"] = sigmoid->readDynamicEnergy;

  f_macroPerf << setw(2) << macroPerf << endl;
  f_reluPerf << setw(2) << reluPerf << endl;
  f_maxpoolPerf << setw(2) << maxpoolPerf << endl;
  f_sigmoidPerf << setw(2) << sigmoidPerf << endl;

  free(param);
  free(tech);
  free(synapticArray);
  free(relu);
  free(maxpool);
  free(sigmoid);
}

} // namespace Refactor
