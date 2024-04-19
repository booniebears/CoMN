/**
 * @file SenseAmp.cpp
 * @author booniebears
 * @brief Sense Amplifier for readout of subarray columns.
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/SenseAmp.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
SenseAmp::SenseAmp(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void SenseAmp::Initialize(int _numOfColumns, bool _isCurrentSense,
                          double _senseVoltage, double _pitchSenseAmp) {
  numOfColumns = _numOfColumns;
  isCurrentSense = _isCurrentSense;
  senseVoltage = _senseVoltage;
  pitchSenseAmp = _pitchSenseAmp;

  if (pitchSenseAmp <= tech->featureSize * 6) {
    throw runtime_error(
        "(SenseAmp.cpp)[Error]: pitch too small, cannot do the layout");
  }
}

void SenseAmp::CalculateArea(double _newHeight, double _newWidth,
                             AreaModify _option) {
  double hSenseP, wSenseP, hSenseN, wSenseN, hSenseIso, wSenseIso, hSenseEn, wSenseEn;
  // Exchange width and height as in the original code
  CalculateGateArea(INV, 1, 0, W_SENSE_P * tech->featureSize, pitchSenseAmp,
                    tech, &wSenseP, &hSenseP);
  CalculateGateArea(INV, 1, 0, W_SENSE_ISO * tech->featureSize, pitchSenseAmp,
                    tech, &wSenseIso, &hSenseIso);
  CalculateGateArea(INV, 1, W_SENSE_N * tech->featureSize, 0, pitchSenseAmp,
                    tech, &wSenseN, &hSenseN);
  CalculateGateArea(INV, 1, W_SENSE_EN * tech->featureSize, 0, pitchSenseAmp,
                    tech, &wSenseEn, &hSenseEn);
  area += (wSenseP * hSenseP) * 2 + (wSenseN * hSenseN) * 2 +
          wSenseIso * hSenseIso + wSenseEn * hSenseEn;
  area *= numOfColumns;

  if (_newWidth && _option == NONE) {
    width = _newWidth;
    height = area / width;
  } else if (_newHeight && _option == NONE) {
    height = _newHeight;
    width = area / height;
  }

  // Capacitance
  capLoad =
      CalculateGateCap((W_SENSE_P + W_SENSE_N) * tech->featureSize, tech) +
      CalculateDrainCap(W_SENSE_N * tech->featureSize, NMOS, pitchSenseAmp, tech) +
      CalculateDrainCap(W_SENSE_P * tech->featureSize, PMOS, pitchSenseAmp, tech) +
      CalculateDrainCap(W_SENSE_ISO * tech->featureSize, PMOS, pitchSenseAmp, tech) +
      CalculateDrainCap(W_SENSE_MUX * tech->featureSize, NMOS, pitchSenseAmp, tech);
}

void SenseAmp::CalculateLatency(double numRead) {
  double gm =
      CalculateTransconductance(W_SENSE_N * tech->featureSize, NMOS, tech) +
      CalculateTransconductance(W_SENSE_P * tech->featureSize, PMOS, tech);
  double tau = capLoad / gm;
  readLatency += tau * log(tech->vdd / senseVoltage);
  readLatency += 1 / param->clkFreq;
  readLatency *= numRead;
}

void SenseAmp::CalculatePower(double numRead) {
  // Leakage
  double idleCurrent =
      CalculateGateLeakage(INV, 1, W_SENSE_EN * tech->featureSize, 0,
                           param->temp, tech) * tech->vdd;
  leakage = idleCurrent * tech->vdd * numOfColumns;
  // Dynamic energy
  readDynamicEnergy = capLoad * tech->vdd * tech->vdd;
  readDynamicEnergy *= numOfColumns * numRead;
}

} // namespace CoMN
