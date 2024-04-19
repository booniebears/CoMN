/**
 * @file Comparator.cpp
 * @author booniebears
 * @brief
 * @date 2023-12-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "../include/Comparator.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {

Comparator::Comparator(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void Comparator::Initialize(int _numOfBits, int _numOfUnits) {
  numOfBits = _numOfBits;
  numOfUnits = _numOfUnits;

  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  widthNand2N = 2 * MIN_NMOS_SIZE * tech->featureSize;
  widthNand2P = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  widthNand3N = 3 * MIN_NMOS_SIZE * tech->featureSize;
  widthNand3P = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
}

void Comparator::CalculateArea(double widthArray) {
  double hInv, wInv, hNand2, wNand2, hNand3, wNand3;
  CalculateGateArea(INV, 1, widthInvN, widthInvP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv,
                    &wInv);
  CalculateGateArea(NAND, 2, widthNand2N, widthNand2P,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand2,
                    &wNand2);
  CalculateGateArea(NAND, 3, widthNand3N, widthNand3P,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand3,
                    &wNand3);
  areaUnit =
      ((hInv * wInv) * 4 + (hNand2 * wNand2) * 4 + (hNand3 * wNand3) * 3) *
      numOfBits;

  area = areaUnit * numOfUnits;
  width = widthArray;
  height = area / width;

  // INV
  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech,
                           &capInvInput, &capInvOutput);
  // NAND2
  CalculateGateCapacitance(NAND, 2, widthNand2N, widthNand2P, hNand2, tech,
                           &capNand2Input, &capNand2Output);
  // NAND3
  CalculateGateCapacitance(NAND, 3, widthNand3N, widthNand3P, hNand3, tech,
                           &capNand3Input, &capNand3Output);
}

void Comparator::CalculateLatency(double _rampInput, double _capLoad,
                                  double numRead) {
  double tr;   /* time constant */
  double gm;   /* transconductance */
  double beta; /* for horowitz calculation */
  double resPullUp, resPullDown;
  double readLatencyIntermediate = 0;
  double rampNand2Output, rampNand3Output;

  // Just use the delay path from Gin to Gout for simplicity
  // 1st bit comparator
  // NAND2
  resPullDown = CalculateOnResistance(widthNand2N, NMOS, param->temp, tech) * 2;
  tr = resPullDown * (capNand2Output + capNand3Input);
  gm = CalculateTransconductance(widthNand2N, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatency += horowitz(tr, beta, _rampInput, &rampNand2Output);
  // NAND3
  resPullUp = CalculateOnResistance(widthNand3P, PMOS, param->temp, tech);
  tr = resPullUp * (capNand3Output + capNand2Input);
  gm = CalculateTransconductance(widthNand3P, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatency += horowitz(tr, beta, rampNand2Output, &rampNand3Output);

  // 2nd bit to the second last bit comparator
  // NAND2
  resPullDown = CalculateOnResistance(widthNand2N, NMOS, param->temp, tech) * 2;
  tr = resPullDown * (capNand2Output + capNand3Input);
  gm = CalculateTransconductance(widthNand2N, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatencyIntermediate +=
      horowitz(tr, beta, rampNand3Output, &rampNand2Output);
  // NAND3
  resPullUp = CalculateOnResistance(widthNand3P, PMOS, param->temp, tech);
  tr = resPullUp * (capNand3Output + capNand2Input);
  gm = CalculateTransconductance(widthNand3P, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatencyIntermediate +=
      horowitz(tr, beta, rampNand2Output, &rampNand3Output);

  readLatency += readLatencyIntermediate * (numOfBits - 2);

  // Last bit comparator
  // NAND2
  resPullDown = CalculateOnResistance(widthNand2N, NMOS, param->temp, tech) * 2;
  tr = resPullDown * (capNand2Output + capNand3Input);
  gm = CalculateTransconductance(widthNand2N, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatency += horowitz(tr, beta, rampNand3Output, &rampNand2Output);
  // NAND3
  resPullUp = CalculateOnResistance(widthNand3P, PMOS, param->temp, tech);
  tr = resPullUp * (capNand3Output + _capLoad);
  gm = CalculateTransconductance(widthNand3P, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatency += horowitz(tr, beta, rampNand2Output, &rampNand3Output);

  readLatency *= numRead;
}

void Comparator::CalculatePower(double numRead) {
  leakage +=
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
      tech->vdd * 4 * numOfBits * numOfUnits;
  leakage += CalculateGateLeakage(NAND, 2, widthNand2N, widthNand2P,
                                  param->temp, tech) *
             tech->vdd * 4 * numOfBits * numOfUnits;
  leakage += CalculateGateLeakage(NAND, 3, widthNand3N, widthNand3P,
                                  param->temp, tech) *
             tech->vdd * 3 * numOfBits * numOfUnits;

  // INV
  readDynamicEnergy +=
      ((capInvInput + capInvOutput) * 4) * tech->vdd * tech->vdd;
  // NAND2
  readDynamicEnergy +=
      ((capNand2Input + capNand2Output) * 4) * tech->vdd * tech->vdd;
  // NAND3
  readDynamicEnergy +=
      ((capNand3Input + capNand3Output) * 3) * tech->vdd * tech->vdd;

  readDynamicEnergy *= numOfBits * numOfUnits * numRead;
}

} // namespace CoMN
