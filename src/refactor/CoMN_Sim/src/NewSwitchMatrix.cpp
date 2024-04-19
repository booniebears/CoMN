/**
 * @file NewSwitchMatrix.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/NewSwitchMatrix.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
NewSwitchMatrix::NewSwitchMatrix(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit(), dff(_param, _technology) {}

void NewSwitchMatrix::Initialize(int _numOfLines, double _activityRowRead) {
  numOfLines = _numOfLines;
  activityRowRead = _activityRowRead;

  dff.Initialize(numOfLines);
  widthTgN = MIN_NMOS_SIZE * tech->featureSize;
  widthTgP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
  resTg = CalculateOnResistance(widthTgN, NMOS, param->temp, tech) *
          LINEAR_REGION_RATIO;
}

void NewSwitchMatrix::CalculateArea(double _newHeight, double _newWidth,
                                    AreaModify _option) {
  double hTg, wTg;
  double minCellHeight = MAX_TRANSISTOR_HEIGHT * tech->featureSize;
  if (_newHeight && _option == NONE) {
    int numOfTgPerCol = _newHeight / minCellHeight;
    int numOfColumnsTg = ceil((double)numOfLines / numOfTgPerCol);
    numOfTgPerCol = ceil((double)numOfLines / numOfColumnsTg);
    double TgHeight = _newHeight / numOfTgPerCol;
    CalculateGateArea(INV, 1, widthTgN, widthTgP, TgHeight, tech, &hTg, &wTg);
    dff.CalculateArea(_newHeight, 0, NONE);
    width = (wTg * 4) * numOfColumnsTg + dff.width;
    height = _newHeight;
  } else {
    CalculateGateArea(INV, 1, widthTgN, widthTgP, minCellHeight, tech, &hTg,
                      &wTg);
    height = hTg * numOfLines;
    dff.CalculateArea(height, 0, NONE);
    width = (wTg * 4) + dff.width;
  }
  area = width * height;
  // Capacitance
  // TG
  capTgGateN = CalculateGateCap(widthTgN, tech);
  capTgGateP = CalculateGateCap(widthTgP, tech);
  CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech, NULL,
                           &capTgDrain);
}

void NewSwitchMatrix::CalculateLatency(double _rampInput, double _capLoad,
                                       double _resLoad, double numRead,
                                       double numWrite) {
  // cout << "********** NewSwitchMatrix::CalculateLatency Begins **********" <<
  // endl;
  double capOutput, tauf;
  capOutput = 5 * capTgDrain;
  tauf = resTg * (capOutput + _capLoad) + _resLoad * _capLoad / 2;
  dff.CalculateLatency(numRead);

  double TgLatency = horowitz(tauf, 0, _rampInput, nullptr);
  readLatency = TgLatency * numRead + dff.readLatency;
  if (numWrite > 0)
    writeLatency = TgLatency * numWrite + dff.readLatency;

  // cout << "dff.readLatency = " << dff.readLatency << endl;
  // cout << "tauf = " << tauf << endl;
  // cout << "resTg = " << resTg << endl;
  // cout << "capOutput = " << capOutput << endl;
  // cout << "capLoad = " << _capLoad << endl;
  // cout << "resLoad = " << _resLoad << endl;
  // cout << "TgLatency = " << TgLatency << endl;
  // cout << "********** NewSwitchMatrix::CalculateLatency Ends **********" <<
  // endl;
}

void NewSwitchMatrix::CalculatePower(double numRead, double numWrite) {
  // cout << "********** NewSwitchMatrix::CalculatePower Begins **********" << endl;
  dff.CalculatePower(numRead, numOfLines);
  leakage = dff.leakage;
  // cout << "dff.leakage = " << dff.leakage << endl;
  // cout << "numRead = " << numRead << endl;
  // cout << "numOfLines = " << numOfLines << endl;

  // 1 TG pass Vaccess to CMOS gate to select the row
  readDynamicEnergy += (capTgDrain * 2) * param->accessVoltage *
                       param->accessVoltage * numOfLines * activityRowRead;
  // 2 TG pass Vread to BL, total loading is 5 Tg Drain capacitance
  readDynamicEnergy += (capTgDrain * 5) * param->readVoltage *
                       param->readVoltage * numOfLines * activityRowRead;
  // open 3 TG when selected
  readDynamicEnergy += (capTgGateN + capTgGateP) * 3 * tech->vdd * tech->vdd *
                       numOfLines * activityRowRead;
  readDynamicEnergy *= numRead;
  readDynamicEnergy += dff.readDynamicEnergy;

  // Write dynamic energy (2-step write and average case half SET and half
  // RESET) 1T1R connect to rows, when writing, pass GND to BL, no transmission
  // energy acrossing BL

  // 1 TG pass Vaccess to CMOS gate to select the row
  writeDynamicEnergy +=
      (capTgDrain * 2) * param->accessVoltage * param->accessVoltage;
  // open 2 TG when Q selected, and *2 means switching from one selected row to
  // another
  writeDynamicEnergy +=
      (capTgGateN + capTgGateP) * 2 * tech->vdd * tech->vdd * 2;
  // always open one TG when writing
  writeDynamicEnergy += (capTgGateN + capTgGateP) * tech->vdd * tech->vdd;
  writeDynamicEnergy *= numWrite;
  if (numWrite != 0) {
    // Use DFF read energy here as writeDynamicEnergy.
    writeDynamicEnergy += dff.readDynamicEnergy;
  }
  // cout << "********** NewSwitchMatrix::CalculatePower Ends **********" << endl;
}

} // namespace CoMN
