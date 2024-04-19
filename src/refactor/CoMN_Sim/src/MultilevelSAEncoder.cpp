/**
 * @file MultilevelSAEncoder.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <cmath>

#include "../include/MultilevelSAEncoder.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
MultilevelSAEncoder::MultilevelSAEncoder(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void MultilevelSAEncoder::Initialize(int _levelOutput, int _numOfEncoders) {
  levelOutput = _levelOutput;
  numOfEncoders = _numOfEncoders;
  numInput = ceil(levelOutput / 2);
  numGate = ceil(log2(levelOutput));

  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
  widthNandN = 2 * MIN_NMOS_SIZE * tech->featureSize;
  widthNandP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
}

void MultilevelSAEncoder::CalculateArea(double _newHeight, double _newWidth,
                                        AreaModify _option) {
  double wEncoder, hEncoder, wNand, hNand, wNandLg, hNandLg, wInv, hInv;
  // NAND2
  CalculateGateArea(NAND, 2, widthNandN, widthNandP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand,
                    &wNand);
  // INV
  CalculateGateArea(INV, 1, widthInvN, widthInvP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv,
                    &wInv);
  // Large NAND in Encoder
  CalculateGateArea(NAND, numInput, widthNandN, widthNandP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNandLg,
                    &wNandLg);
  wEncoder = 2 * wInv + wNand + wNandLg;
  hEncoder = max((levelOutput - 1) * hInv, (levelOutput - 1) * hNand);
  if (_newWidth && _option == NONE) {
    int numOfEncodersPerRow = _newWidth / wEncoder;
    if (numOfEncodersPerRow > numOfEncoders) {
      numOfEncodersPerRow = numOfEncoders;
    }
    int numOfRowsEncoder = ceil((double)numOfEncoders / numOfEncodersPerRow);
    width = max(_newWidth, wEncoder);
    height = hEncoder * numOfRowsEncoder;
  } else { // _newHeight
    int numOfEncodersPerCol = _newHeight / hEncoder;
    if (numOfEncodersPerCol > numOfEncoders) {
      numOfEncodersPerCol = numOfEncoders;
    }
    int numOfColumnsEncoder = ceil((double)numOfEncoders / numOfEncodersPerCol);
    width = wEncoder * numOfColumnsEncoder;
    height = max(_newHeight, hEncoder);
  }
  area = width * height;

  // Capacitance
  // INV
  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech,
                           &capInvInput, &capInvOutput);
  // NAND2
  CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech,
                           &capNandInput, &capNandOutput);
  // Large NAND in Encoder
  CalculateGateCapacitance(NAND, numInput, widthNandN, widthNandP, hNandLg,
                           tech, &capNandLgInput, &capNandLgOutput);
}
void MultilevelSAEncoder::CalculateLatency(double _rampInput, double numRead) {
  double gm;   /* transconductance */
  double beta; /* for horowitz calculation */
  double resPullUp, resPullDown;
  double readLatencyIntermediate = 0;
  double tauf;
  double ramp[8];
  ramp[0] = _rampInput;

  // 1st INV to NAND2
  resPullDown = CalculateOnResistance(widthInvN, NMOS, param->temp, tech) * 2;
  tauf = resPullDown * (capInvOutput + capNandInput * 2);
  gm = CalculateTransconductance(widthNandN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatency += horowitz(tauf, beta, ramp[0], &ramp[1]);

  // 2nd NAND2 to Large NAND
  resPullUp = CalculateOnResistance(widthNandP, PMOS, param->temp, tech);
  tauf = resPullUp * (capNandOutput + capNandLgInput * numInput);
  gm = CalculateTransconductance(widthNandP, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatency += horowitz(tauf, beta, ramp[1], &ramp[2]);

  // 3rd large NAND to INV
  resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech) * 2;
  tauf = resPullDown * (capNandLgOutput + capInvInput);
  gm = CalculateTransconductance(widthNandN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatencyIntermediate += horowitz(tauf, beta, ramp[2], &ramp[3]);

  // 4th INV
  resPullUp = CalculateOnResistance(widthInvP, PMOS, param->temp, tech);
  tauf = resPullUp * capInvOutput;
  gm = CalculateTransconductance(widthNandP, PMOS, tech);
  beta = 1 / (resPullUp * gm);
  readLatencyIntermediate += horowitz(tauf, beta, ramp[3], &ramp[4]);

  readLatency *= numRead;
  // rampOutput = ramp[4];
}

void MultilevelSAEncoder::CalculatePower(double numRead) {
  leakage =
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
          tech->vdd * (levelOutput + numGate) * numOfEncoders +
      CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, param->temp, tech) *
          tech->vdd * (levelOutput + numGate) * numOfEncoders +
      CalculateGateLeakage(NAND, numInput, widthNandN, widthNandP, param->temp,
                           tech) *
          tech->vdd * numGate * numOfEncoders;

  readDynamicEnergy += (capInvInput + capInvOutput) * tech->vdd * tech->vdd *
                       (levelOutput + numGate) * numOfEncoders;
  readDynamicEnergy += (capNandInput + capNandOutput) * tech->vdd * tech->vdd *
                       (levelOutput + numGate) * numOfEncoders;
  readDynamicEnergy += (capNandLgInput + capNandLgOutput) * tech->vdd *
                       tech->vdd * numGate * numOfEncoders;
  readDynamicEnergy *= numRead;
}

} // namespace CoMN
