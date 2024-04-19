/**
 * @file DecoderDriver.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/DecoderDriver.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
DecoderDriver::DecoderDriver(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void DecoderDriver::Initialize(RowColMode _mode, int _numOfRows,
                               int _numOfColumns, double resMemCellOn) {
  mode = _mode;
  numOfRows = _numOfRows;
  numOfColumns = _numOfColumns;

  // INV
  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  // TG
  resTg = resMemCellOn / numOfColumns * IR_DROP_TOLERANCE;
  widthTgN = CalculateOnResistance(tech->featureSize, NMOS, param->temp, tech) *
             tech->featureSize / (resTg * 2);
  widthTgP = CalculateOnResistance(tech->featureSize, PMOS, param->temp, tech) *
             tech->featureSize / (resTg * 2);
}

void DecoderDriver::CalculateArea(double _newHeight, double _newWidth,
                                  AreaModify _option) {
  double hInv, wInv, hTg, wTg;
  double minCellHeight = MAX_TRANSISTOR_HEIGHT * tech->featureSize;
  double minCellWidth =
      2 * (POLY_WIDTH + MIN_GAP_BET_GATE_POLY) * tech->featureSize;
  if (_newHeight && _option == NONE) {
    int numOfTgPerCol = _newHeight / minCellHeight;
    int numOfColumnsTg = ceil((double)numOfRows / numOfTgPerCol);
    numOfTgPerCol = ceil((double)numOfRows / numOfColumnsTg);
    double TgHeight = _newHeight / numOfTgPerCol;
    CalculateGateArea(INV, 1, widthTgN, widthTgP, TgHeight, tech, &hTg, &wTg);
  } else if (_newWidth && _option == NONE) {
    int numOfTgPerRow = _newWidth / minCellWidth;
    int numOfRowsTg = ceil((double)numOfRows / numOfTgPerRow);
    numOfTgPerRow = ceil((double)numOfRows / numOfRowsTg);
    double TgWidth = _newWidth / numOfTgPerRow;
    int numFold = (int)(TgWidth / (0.5 * minCellWidth)) - 1;
    CalculatePassGateArea(widthTgN, widthTgP, tech, numFold, &hTg, &wTg);
  } else {
    CalculateGateArea(INV, 1, widthTgN, widthTgP, minCellHeight, tech, &hTg,
                      &wTg);
  }
  // INV
  CalculateGateArea(INV, 1, widthInvN, widthInvP, minCellHeight, tech, &hInv,
                    &wInv);
  double hUnit, wUnit;
  if (param->accessType == CMOS_access) { // 1T1R
    if (mode == ROW_MODE) {
      hUnit = max(hInv, hTg);
      wUnit = wInv + wTg * 3;
    } else {
      hUnit = hInv + hTg * 3;
      wUnit = max(wInv, wTg);
    }
  } else { // Cross-point
    if (mode == ROW_MODE) {
      hUnit = max(hInv, hTg);
      wUnit = wInv + wTg * 2;
    } else {
      hUnit = hInv + hTg * 2;
      wUnit = max(wInv, wTg);
    }
  }

  if (mode == ROW_MODE) { // Connect to rows
    if (_newHeight && _option == NONE) {
      int numUnitPerCol = (int)(_newHeight / hUnit);
      int numColUnit = (int)ceil((double)numOfRows / numUnitPerCol);
      if (numColUnit > numOfRows) {
        numColUnit = numOfRows;
      }
      height = _newHeight;
      width = wUnit * numColUnit;
    } else {
      height = hUnit * numOfRows;
      width = wUnit;
    }
  } else { // Connect to columns
    if (_newWidth && _option == NONE) {
      int numRowUnit, numUnitPerRow;
      numUnitPerRow = (int)(_newWidth / wUnit);
      numRowUnit = (int)ceil((double)numOfRows / numUnitPerRow);
      if (numRowUnit > numOfRows) {
        numRowUnit = numOfRows;
      }
      height = hUnit * numRowUnit;
      width = _newWidth;
    } else {
      height = hUnit;
      width = wUnit * numOfRows;
    }
  }
  area = height * width;

  // Capacitance
  // INV
  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech,
                           &capInvInput, &capInvOutput);
  // TG
  capTgGateN = CalculateGateCap(widthTgN, tech);
  capTgGateP = CalculateGateCap(widthTgP, tech);
  CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech, NULL,
                           &capTgDrain);
}

void DecoderDriver::CalculateLatency(double _rampInput, double _capLoad,
                                     double _resLoad, double numRead,
                                     double numWrite) {
  double capOutput;
  double tauf; /* time constant */
  double gm;   /* transconductance */
  double beta; /* for horowitz calculation */

  capOutput = capTgDrain * 2;
  tauf = resTg * (capOutput + _capLoad) + _resLoad * _capLoad / 2;
  // TODO: In horowitz func, beta is fixed to 0.5. Why??
  readLatency = horowitz(tauf, 0, _rampInput, nullptr) * numRead;
  writeLatency = horowitz(tauf, 0, _rampInput, nullptr) * numWrite;
}

void DecoderDriver::CalculatePower(int numReadCells, int numWriteCells,
                                   double numRead, double numWrite) {
  // numReadCells = ceil((double)numOfColumns / numOfColumnsMux)
  // numWriteCells = numOfColumns
  leakage +=
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
      tech->vdd * numOfRows;
  if (param->accessType == CMOS_access) {
    // Selected SLs and BLs are floating
    // Unselected SLs and BLs are GND
    readDynamicEnergy += (capInvInput + capTgGateN * 2 + capTgGateP) *
                         tech->vdd * tech->vdd * numReadCells;
    readDynamicEnergy += (capInvOutput + capTgGateP * 2 + capTgGateN) *
                         tech->vdd * tech->vdd * numReadCells;
  } else {
    // For WL decoder driver, the selected WLs are GND
    // For BL decoder driver, the selected BLs are floating
    // No matter which one, the unselected WLs/BLs are read voltage
    readDynamicEnergy += (capInvInput + capTgGateN + capTgGateP) * tech->vdd *
                         tech->vdd * numReadCells;
    readDynamicEnergy += (capInvOutput + capTgGateP + capTgGateN) * tech->vdd *
                         tech->vdd * numReadCells;
    readDynamicEnergy += (capTgDrain * 2) * param->readVoltage *
                         param->readVoltage * (numOfRows - numReadCells);
  }
  readDynamicEnergy *= numRead;

  if (param->accessType == CMOS_access) {
    writeDynamicEnergy += (capInvInput + capTgGateN * 2 + capTgGateP) *
                          tech->vdd * tech->vdd * numWriteCells;
    writeDynamicEnergy += (capInvOutput + capTgGateP * 2 + capTgGateN) *
                          tech->vdd * tech->vdd * numWriteCells;
    writeDynamicEnergy += (capTgDrain * 2) * param->writeVoltage *
                          param->writeVoltage * numWriteCells;
  } else {
    writeDynamicEnergy += (capInvInput + capTgGateN + capTgGateP) * tech->vdd *
                          tech->vdd * numWriteCells;
    writeDynamicEnergy += (capInvOutput + capTgGateP + capTgGateN) * tech->vdd *
                          tech->vdd * numWriteCells;
    if (mode == ROW_MODE) { // Connects to rows
      writeDynamicEnergy += (capTgDrain * 2) * param->writeVoltage *
                            param->writeVoltage * numWriteCells;
      writeDynamicEnergy += (capTgDrain * 2) * param->writeVoltage / 2 *
                            param->writeVoltage / 2 *
                            (numOfRows - numWriteCells);
    } else { // Connects to columns
      writeDynamicEnergy += (capTgDrain * 2) * param->writeVoltage / 2 *
                            param->writeVoltage / 2 *
                            (numOfRows - numWriteCells);
    }
    writeDynamicEnergy *= numWrite;
  }
}

} // namespace CoMN
