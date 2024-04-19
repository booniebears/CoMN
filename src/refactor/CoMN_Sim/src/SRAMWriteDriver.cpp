/**
 * @file SRAMWriteDriver.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <iostream>

#include "../include/SRAMWriteDriver.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
SRAMWriteDriver::SRAMWriteDriver(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void SRAMWriteDriver::Initialize(int _numOfLines, double _activityColWrite) {
  numOfLines = _numOfLines;
  activityColWrite = _activityColWrite;

  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
}

void SRAMWriteDriver::CalculateArea(double _newHeight, double _newWidth,
                                    AreaModify _option) {
  double hInv, wInv;
  CalculateGateArea(INV, 1, widthInvN, widthInvP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv,
                    &wInv);
  double hUnit = hInv * 3, wUnit = wInv;
  if (_newWidth && _option == NONE) {
    int numOfUnitsPerLine = _newWidth / wUnit;
    if (numOfUnitsPerLine > numOfLines)
      numOfUnitsPerLine = numOfLines;
    if (wUnit > _newWidth) {
      throw runtime_error(
          "SRAMWriteDriver width is even larger than the assigned width!!");
    }
    int numOfLinesUnit = ceil((double)numOfLines / numOfUnitsPerLine);
    width = _newWidth;
    height = numOfLinesUnit * hUnit;
  } else {
    width = numOfLines * wUnit;
    height = hUnit;
  }
  area = width * height;

  // Capacitance
  // INV
  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech,
                           &capInvInput, &capInvOutput);
}

void SRAMWriteDriver::CalculateLatency(double _rampInput, double _capLoad,
                                       double _resLoad, double numWrite) {
  double gm;   /* transconductance */
  double beta; /* for horowitz calculation */
  double resPullUp, resPullDown;
  double tauf;
  double rampInvOutput;

  // 1st stage INV (Pullup)
  resPullUp = CalculateOnResistance(widthInvP, PMOS, param->temp, tech);
  tauf = resPullUp * (capInvOutput + capInvInput);
  gm = CalculateTransconductance(widthInvP, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  writeLatency += horowitz(tauf, beta, _rampInput, &rampInvOutput);

  // 2nd stage INV (Pulldown)
  resPullDown = CalculateOnResistance(widthInvN, NMOS, param->temp, tech);
  tauf = resPullDown * (_capLoad + capInvOutput) + _resLoad * _capLoad / 2;
  gm = CalculateTransconductance(widthInvN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  writeLatency += horowitz(tauf, beta, rampInvOutput, nullptr);
  writeLatency *= numWrite;
  // For SRAM Write Driver, readLatency = 0;
}

void SRAMWriteDriver::CalculatePower(double numWrite) {
  //
  leakage =
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
      tech->vdd * 3;
  writeDynamicEnergy = (capInvInput + capInvOutput) * tech->vdd * tech->vdd *
                       numOfLines * activityColWrite;
  writeDynamicEnergy *= numWrite;
}

} // namespace CoMN
