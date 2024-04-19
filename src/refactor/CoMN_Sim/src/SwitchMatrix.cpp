/**
 * @file SwitchMatrix.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-18
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/SwitchMatrix.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
SwitchMatrix::SwitchMatrix(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit(), dff(_param, _technology) {}

// TODO: By default, the switch matrix is in neuro mode rather than memory mode;
// and no parallelWrite. "write" related members not necessary in training.
void SwitchMatrix::Initialize(RowColMode _mode, int _numOfLines, double _resTg,
                              double _activityRowRead, double _activityColWrite,
                              double _numOfWritePulses) {
  mode = _mode;
  numOfLines = _numOfLines;
  resTg = _resTg * IR_DROP_TOLERANCE;
  activityRowRead = _activityRowRead;
  activityColWrite = _activityColWrite;
  numOfWritePulses = _numOfWritePulses;

  dff.Initialize(numOfLines); // TODO: why?

  widthTgN = CalculateOnResistance(tech->featureSize, NMOS, param->temp, tech) *
             tech->featureSize * LINEAR_REGION_RATIO / (resTg * 2);
  widthTgP = CalculateOnResistance(tech->featureSize, PMOS, param->temp, tech) *
             tech->featureSize * LINEAR_REGION_RATIO / (resTg * 2);
}

void SwitchMatrix::CalculateArea(double _newHeight, double _newWidth,
                                 AreaModify _option) {
  // cout << "********** SwitchMatrix::CalculateArea Begins **********" << endl;
  double hTg, wTg;
  if (mode == ROW_MODE) {
    double minCellHeight = MAX_TRANSISTOR_HEIGHT * tech->featureSize;
    if (_newHeight && _option == NONE) {
      if (_newHeight < minCellHeight) {
        throw runtime_error("(SwitchMatrix.cpp)[Error]: pass gate height is "
                            "even larger than the array height ");
      }
      int numOfTgPairPerCol = _newHeight / minCellHeight;
      int numOfColTgPair = ceil((double)numOfLines / numOfTgPairPerCol);
      numOfTgPairPerCol = ceil((double)numOfLines / numOfColTgPair);
      double TgHeight = _newHeight / numOfTgPairPerCol;
      CalculateGateArea(INV, 1, widthTgN, widthTgP, TgHeight, tech, &hTg, &wTg);

      dff.CalculateArea(_newHeight, 0, NONE);
      height = _newHeight;
      width = (wTg * 2) * numOfColTgPair + dff.width;
    } else {
      CalculateGateArea(INV, 1, widthTgN, widthTgP, minCellHeight, tech, &hTg,
                        &wTg); // Pass gate with folding
      height = hTg * numOfLines;
      dff.CalculateArea(height, 0, NONE);
      width = (wTg * 2) + dff.width;
    }
  } else { // REGULAR_COL
    double minCellWidth = 2 * (POLY_WIDTH + MIN_GAP_BET_GATE_POLY) *
                          tech->featureSize; // min standard cell width for 1 Tg
    if (_newWidth && _option == NONE) {
      int numOfTgPairPerRow = _newWidth / (minCellWidth * 2);
      int numOfRowTgPair = ceil((double)numOfLines / numOfTgPairPerRow);
      numOfTgPairPerRow = ceil((double)numOfLines / numOfRowTgPair);
      double TgWidth = _newWidth / (numOfTgPairPerRow * 2);
      // get the max number of folding
      int numFold = (int)(TgWidth / (0.5 * minCellWidth)) - 1;
      // widthTgN, widthTgP and numFold can determine the height and width of
      // each pass gate
      CalculatePassGateArea(widthTgN, widthTgP, tech, numFold, &hTg, &wTg);
      // DFF
      dff.CalculateArea(0, _newWidth, NONE);

      width = _newWidth;
      height = hTg * numOfRowTgPair + dff.height;
      // cout << "minCellWidth = " << minCellWidth << endl;
      // cout << "TgWidth = " << TgWidth << endl;
      // cout << "numTgPairPerRow = " << numOfTgPairPerRow << endl;
      // cout << "numFold = " << numFold << endl;
    } else {
      // Default (pass gate with folding=1)
      CalculatePassGateArea(widthTgN, widthTgP, tech, 1, &hTg, &wTg);
      width = wTg * 2 * numOfLines;
      dff.CalculateArea(0, width, NONE);
      height = hTg + dff.height;
    }
  }
  area = height * width;

  // Capacitance
  // TG
  capTgGateN = CalculateGateCap(widthTgN, tech);
  capTgGateP = CalculateGateCap(widthTgP, tech);
  CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech, NULL,
                           &capTgDrain);
  // cout << "********** SwitchMatrix::CalculateArea Ends **********" << endl;
}
void SwitchMatrix::CalculateLatency(double _rampInput, double _capLoad,
                                    double _resLoad, double numRead,
                                    double numWrite) {
  double capOutput = capTgDrain * 3;
  double tauf = resTg * (capOutput + _capLoad) + _resLoad * _capLoad / 2;

  dff.CalculateLatency(numRead); // numRead is used here.
  // TG, TODO: beta = 0??
  double TgLatency = horowitz(tauf, 0, _rampInput, &rampOutput);
  readLatency = dff.readLatency + TgLatency * numRead;
  // use dff.readLatency to represent
  if (numWrite > 0)
    writeLatency = dff.readLatency + TgLatency * numWrite;
}

void SwitchMatrix::CalculatePower(double numRead, double numWrite) {
  dff.CalculatePower(numRead, numOfLines);
  leakage = dff.leakage;

  if (mode == ROW_MODE) {
    readDynamicEnergy += (capTgDrain * 3) * param->readVoltage *
                         param->readVoltage * numOfLines * activityRowRead;
    readDynamicEnergy += (capTgGateN + capTgGateP) * tech->vdd * tech->vdd *
                         numOfLines * activityRowRead;
  }
  // No read energy in COL_MODE
  readDynamicEnergy *= numRead;
  readDynamicEnergy += dff.readDynamicEnergy;

  if (param->accessType == CMOS_access) {
    if (mode == ROW_MODE) {
      // Selected row in LTP, *2 means switching from one selected row to
      // another
      writeDynamicEnergy +=
          (capTgGateN + capTgGateP) * tech->vdd * tech->vdd * 2;
    } else {
      // LTP
      // Selected columns
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage *
                            param->writeVoltage * numOfWritePulses *
                            numOfLines * activityColWrite / 2;
      // Unselected columns
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage *
                            param->writeVoltage *
                            (numOfLines - numOfLines * activityColWrite / 2);
      // LTD
      // Selected columns
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage *
                            param->writeVoltage * numOfWritePulses *
                            numOfLines * activityColWrite / 2;

      writeDynamicEnergy +=
          (capTgGateN + capTgGateP) * tech->vdd * tech->vdd * numOfLines;
    }
  } else { // Cross-point
    if (mode == ROW_MODE) {
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage *
                            param->writeVoltage; // Selected row in LTP
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage / 2 *
                            param->writeVoltage / 2 *
                            (numOfLines - 1); // Unselected rows in LTP and LTD
      writeDynamicEnergy +=
          (capTgGateN + capTgGateP) * tech->vdd * tech->vdd * numOfLines;
    } else {
      // Selected columns in LTP
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage *
                            param->writeVoltage * numOfWritePulses *
                            numOfLines * activityColWrite / 2;
      // Selected columns in LTD
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage *
                            param->writeVoltage * numOfWritePulses *
                            numOfLines * activityColWrite / 2;
      // Total unselected columns in LTP and LTD within the 2-step write
      writeDynamicEnergy += (capTgDrain * 3) * param->writeVoltage / 2 *
                            param->writeVoltage / 2 * numOfLines;
      writeDynamicEnergy +=
          (capTgGateN + capTgGateP) * tech->vdd * tech->vdd * numOfLines;
    }
  }
  writeDynamicEnergy *= numWrite;
  writeDynamicEnergy += dff.writeDynamicEnergy;
}

} // namespace CoMN
