/**
 * @file NewDecoderDriver.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/NewDecoderDriver.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
NewDecoderDriver::NewDecoderDriver(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void NewDecoderDriver::Initialize(int _numOfRows) {
  numOfRows = _numOfRows;
  // NAND2
  widthNandN = 2 * MIN_NMOS_SIZE * tech->featureSize;
  widthNandP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  // INV
  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  // Transmission Gate
  widthTgN = MIN_NMOS_SIZE * tech->featureSize;
  widthTgP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
}

void NewDecoderDriver::CalculateArea(double _newHeight, double _newWidth,
                                     AreaModify _option) {
  double hNand, wNand, hInv, wInv, hTg, wTg;
  double minCellHeight = MAX_TRANSISTOR_HEIGHT * tech->featureSize;

  // NAND2
  CalculateGateArea(NAND, 2, widthNandN, widthNandP, minCellHeight, tech,
                    &hNand, &wNand);
  // INV
  CalculateGateArea(INV, 1, widthInvN, widthInvP, minCellHeight, tech, &hInv,
                    &wInv);
  // TG
  CalculateGateArea(INV, 1, widthTgN, widthTgP, minCellHeight, tech, &hTg,
                    &wTg);
  if (_newHeight && _option == NONE) {
    double TgHeight = _newHeight / numOfRows;
    if (TgHeight < minCellHeight) {
      // TODO: How do the calculations work?
      int numOfColumnsTg = ceil(minCellHeight / TgHeight);
      TgHeight = _newHeight / (int)ceil((double)numOfRows / numOfColumnsTg);
    }
    CalculateGateArea(INV, 1, widthTgN, widthTgP, TgHeight, tech, &hTg, &wTg);
    double hUnit = max(max(hInv, hNand), hTg);
    double wUnit = 3 * wNand + wInv + 2 * wTg;
    int numOfUnitsPerCol = _newHeight / hUnit;
    if (numOfUnitsPerCol > numOfRows) {
      numOfUnitsPerCol = numOfRows;
    }
    int numOfColumnsUnit = ceil((double)numOfRows / numOfUnitsPerCol);
    width = (3 * wNand + wInv + 2 * wTg) * numOfColumnsUnit;
    height = _newHeight;
  } else { // MAGIC/OVERRIDE
    height = max(max(hInv, hNand), hTg) * numOfRows;
    width = 3 * wNand + wInv + 2 * wTg;
  }
  area = width * height;
  // Resistance
  // TG
  double resTgN, resTgP;
  resTgN = CalculateOnResistance(widthTgN, NMOS, param->temp, tech);
  resTgP = CalculateOnResistance(widthTgP, PMOS, param->temp, tech);
  resTg = 1 / (1 / resTgN + 1 / resTgP);

  // Capacitance
  // NAND2
  CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech,
                           &capNandInput, &capNandOutput);
  // INV
  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech,
                           &capInvInput, &capInvOutput);
  // TG
  capTgGateN = CalculateGateCap(widthTgN, tech);
  capTgGateP = CalculateGateCap(widthTgP, tech);
  CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech, nullptr,
                           &capTgDrain);
}

void NewDecoderDriver::CalculateLatency(double _rampInput, double _capLoad,
                                        double _resLoad, double numRead,
                                        double numWrite) {
  double resPullDown; // NAND2 pulldown resistance
  double resPullUp;   // INV pullup resistance
  double capOutput;
  double taufNand; /* NAND time constant */
  double gmNand;   /* NAND transconductance */
  double betaNand; /* for NAND horowitz calculation */
  double taufInv;  /* INV time constant */
  double gmInv;    /* INV transconductance */
  double betaInv;  /* for INV horowitz calculation */
  double trTg;     /* TG time constant */

  // 1st stage: NAND2
  resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech) *
                2; // pulldown 2 NMOS in series
  taufNand = resPullDown * (capNandOutput + capInvInput); // connect to INV
  gmNand = CalculateTransconductance(widthNandN, NMOS, tech);
  betaNand = 1 / (resPullDown * gmNand);
  readLatency += horowitz(taufNand, betaNand, _rampInput, nullptr);
  writeLatency += horowitz(taufNand, betaNand, _rampInput, nullptr);

  // 2ed stage: INV
  resPullUp = CalculateOnResistance(widthInvP, PMOS, param->temp, tech);
  taufInv =
      resPullUp * (capInvOutput + 2 * capNandInput); // connect to 2 NAND2 gate
  gmInv = CalculateTransconductance(widthNandP, PMOS, tech);
  betaInv = 1 / (resPullUp * gmInv);
  readLatency += horowitz(taufInv, betaInv, _rampInput, nullptr);
  writeLatency += horowitz(taufInv, betaInv, _rampInput, nullptr);

  // 3ed stage: NAND2
  resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech) * 2;
  taufNand = resPullDown * (capNandOutput + capTgGateP +
                            capTgGateN); // connect to 2 transmission gates
  gmNand = CalculateTransconductance(widthNandN, NMOS, tech);
  betaNand = 1 / (resPullDown * gmNand);
  readLatency += horowitz(taufNand, betaNand, _rampInput, nullptr);
  writeLatency += horowitz(taufNand, betaNand, _rampInput, nullptr);

  // 4th stage: TG
  capOutput = 2 * capTgDrain;
  trTg = resTg * (capOutput + _capLoad) + _resLoad * _capLoad / 2;
  // get from chargeLatency in the original SubArray.cpp
  readLatency += horowitz(trTg, 0, 1e20, nullptr);
  writeLatency += horowitz(trTg, 0, 1e20, nullptr);

  readLatency *= numRead;
  writeLatency *= numWrite;
}

void NewDecoderDriver::CalculatePower(double numRead, double numWrite) {
  // NAND2
  leakage +=
      CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, param->temp, tech) *
      tech->vdd * numOfRows * 2;
  // INV
  leakage +=
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
      tech->vdd * numOfRows * 2;
  // Assume no leakage in Tg

  // Read dynamic energy (only one row activated)
  // NAND2 input charging ( 0 to 1 )
  readDynamicEnergy += capNandInput * tech->vdd * tech->vdd;
  // INV output charging ( 0 to 1 )
  readDynamicEnergy += (capInvOutput + capTgGateN) * tech->vdd * tech->vdd;
  // NAND2 output charging ( 0 to 1 )
  readDynamicEnergy +=
      (capNandOutput + capTgGateN + capTgGateP) * tech->vdd * tech->vdd;
  // TG gate energy
  readDynamicEnergy += capTgDrain * param->readVoltage * param->readVoltage;
  readDynamicEnergy *= numRead;

  // Write dynamic energy (only one row activated)
  writeDynamicEnergy += capNandInput * tech->vdd * tech->vdd;
  writeDynamicEnergy += (capInvOutput + capTgGateN) * tech->vdd * tech->vdd;
  writeDynamicEnergy +=
      (capNandOutput + capTgGateN + capTgGateP) * tech->vdd * tech->vdd;
  writeDynamicEnergy += capTgDrain * param->writeVoltage * param->writeVoltage;
  writeDynamicEnergy *= numWrite;
}

} // namespace CoMN
