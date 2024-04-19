/**
 * @file Adder.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/Adder.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
Adder::Adder(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void Adder::Initialize(int _numOfBits, int _numOfAdders) {
  numOfBits = _numOfBits;
  numOfAdders = _numOfAdders;
  widthNandN = 2 * MIN_NMOS_SIZE * tech->featureSize;
  widthNandP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
}

void Adder::CalculateArea(double _newHeight, double _newWidth,
                          AreaModify _option) {
  double hNand, wNand;
  double hAdder, wAdder;
  CalculateGateArea(NAND, 2, widthNandN, widthNandP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand,
                    &wNand);
  if (_newWidth && _option == NONE) {
    hAdder = hNand * numOfBits;
    wAdder = wNand * 9;
    if (wAdder > _newWidth) {
      throw runtime_error(
          "(Adder.cpp)[Error]: wAdder greater than assigned width!!!");
    }
    int numOfAddersPerRow = _newWidth / wAdder;
    if (numOfAddersPerRow > numOfAdders)
      numOfAddersPerRow = numOfAdders;
    int numOfRowsAdder = ceil(double(numOfAdders) / numOfAddersPerRow);
    width = _newWidth;
    height = hAdder * numOfRowsAdder;
  } else if (_newHeight && _option == NONE) {
    hAdder = hNand;
    wAdder = wNand * 9 * numOfBits;
    if (hAdder > _newHeight) {
      throw runtime_error(
          "(Adder.cpp)[Error]: hAdder greater than assigned height!!!");
    }
    int numOfAddersPerCol = _newHeight / hAdder;
    if (numOfAddersPerCol > numOfAdders) {
      numOfAddersPerCol = numOfAdders;
    }
    int numOfColumnsAdder = ceil((double)numOfAdders / numOfAddersPerCol);
    width = wAdder * numOfColumnsAdder;
    height = _newHeight;
  } else { // One row of Adders by default
    hAdder = hNand;
    wAdder = wNand * 9 * numOfBits;
    width = wAdder * numOfAdders;
    height = hAdder;
  }
  area = height * width;

  // NAND2 capacitance
  CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech,
                           &capNandInput, &capNandOutput);
}

void Adder::CalculateLatency(double _rampInput, double _capLoad,
                             double numRead) {
  double tr;   /* time constant */
  double gm;   /* transconductance */
  double beta; /* for horowitz calculation */
  double resPullUp, resPullDown;
  double readLatencyIntermediate = 0;
  double ramp[8];
  ramp[0] = _rampInput;

  // Calibration data pattern is A=1111111..., B=1000000... and Cin=1
  // 1st
  resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech) * 2;
  tr = resPullDown * (capNandOutput + capNandInput * 3);
  gm = CalculateTransconductance(widthNandN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatency += horowitz(tr, beta, ramp[0], &ramp[1]);

  // 2nd
  resPullUp = CalculateOnResistance(widthNandP, PMOS, param->temp, tech);
  tr = resPullUp * (capNandOutput + capNandInput * 2);
  gm = CalculateTransconductance(widthNandP, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatency += horowitz(tr, beta, ramp[1], &ramp[2]);

  // 3rd
  resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech) * 2;
  tr = resPullDown * (capNandOutput + capNandInput * 3);
  gm = CalculateTransconductance(widthNandN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatencyIntermediate += horowitz(tr, beta, ramp[2], &ramp[3]);

  // 4th
  resPullUp = CalculateOnResistance(widthNandP, PMOS, param->temp, tech);
  tr = resPullUp * (capNandOutput + capNandInput * 2);
  gm = CalculateTransconductance(widthNandP, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatencyIntermediate += horowitz(tr, beta, ramp[3], &ramp[4]);

  if (numOfBits > 2) {
    readLatency += readLatencyIntermediate * (numOfBits - 2);
  }

  // 5th
  resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech) * 2;
  tr = resPullDown * (capNandOutput + capNandInput * 3);
  gm = CalculateTransconductance(widthNandN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatency += horowitz(tr, beta, ramp[4], &ramp[5]);

  // 6th
  resPullUp = CalculateOnResistance(widthNandP, PMOS, param->temp, tech);
  tr = resPullUp * (capNandOutput + capNandInput);
  gm = CalculateTransconductance(widthNandP, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatency += horowitz(tr, beta, ramp[5], &ramp[6]);

  // 7th
  resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech) * 2;
  tr = resPullDown * (capNandOutput + _capLoad);
  gm = CalculateTransconductance(widthNandN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatency += horowitz(tr, beta, ramp[6], &ramp[7]);

  readLatency *= numRead;
  // rampOutput = ramp[7];
}

void Adder::CalculatePower(double numRead, int numAdderPerOperation) {
  leakage =
      CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, param->temp, tech) *
      tech->vdd * 9 * numOfBits * numOfAdders;

  // First stage
  readDynamicEnergy +=
      (capNandInput * 6) * tech->vdd * tech->vdd; // Input of 1 and 2 and Cin
  readDynamicEnergy +=
      (capNandOutput * 2) * tech->vdd * tech->vdd; // Output of S[0] and 5
  // Second and later stages
  readDynamicEnergy +=
      (capNandInput * 7) * tech->vdd * tech->vdd * (numOfBits - 1);
  readDynamicEnergy +=
      (capNandOutput * 3) * tech->vdd * tech->vdd * (numOfBits - 1);

  // Hidden transition
  // First stage
  readDynamicEnergy +=
      (capNandOutput + capNandInput) * tech->vdd * tech->vdd * 2; // #2 and #3
  readDynamicEnergy +=
      (capNandOutput + capNandInput * 2) * tech->vdd * tech->vdd; // #4
  readDynamicEnergy +=
      (capNandOutput + capNandInput * 3) * tech->vdd * tech->vdd; // #5
  readDynamicEnergy +=
      (capNandOutput + capNandInput) * tech->vdd * tech->vdd; // #6
  // Second and later stages
  readDynamicEnergy += (capNandOutput + capNandInput * 3) * tech->vdd *
                       tech->vdd * (numOfBits - 1); // # 1
  readDynamicEnergy += (capNandOutput + capNandInput) * tech->vdd * tech->vdd *
                       (numOfBits - 1); // # 3
  readDynamicEnergy += (capNandOutput + capNandInput) * tech->vdd * tech->vdd *
                       2 * (numOfBits - 1); // #6 and #7

  readDynamicEnergy *= min(numAdderPerOperation, numOfAdders) * numRead;
}

} // namespace CoMN
