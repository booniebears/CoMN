/**
 * @file PE.h
 * @author booniebears
 * @brief
 * @date 2023-11-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PE_H_
#define PE_H_

#include "json.hpp"
#include <fstream>
#include <vector>

using namespace std;
using json = nlohmann::json;

namespace Refactor {

struct PEInfo {
  vector<int> Subarray;
  double MaxConductance, MinConductance;
  int ADCNum, ADCLevel; // ADCNum: How many ADCs in a macro
  double ADC_power, ADC_fre, ADC_area;
};

struct CoMNInfo {
  int technode, numRowSubArray, numColSubArray, levelOutput, numColMuxed;
  double featuresize, readPulseWidth, readVoltage, resistanceOn, resistanceOff,
      ADC_power, ADC_delay, ADC_area;
  bool user_defined;
};

void PE_core_energy(PEInfo &info, int featureSize);

void CoMN_interface(CoMNInfo &info);

} // namespace Refactor

#endif // !PE_H_
