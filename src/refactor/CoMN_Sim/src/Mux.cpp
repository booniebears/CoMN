/**
 * @file Mux.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "../include/Mux.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
Mux::Mux(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

// TODO: All Analog Mux??
void Mux::Initialize(int _numOfMux, int _numOfInput, double _resTg) {
  numOfMux = _numOfMux;
  numOfInput = _numOfInput;
  resTg = _resTg * IR_DROP_TOLERANCE;
  widthTgN = CalculateOnResistance(tech->featureSize, NMOS, param->temp, tech) *
             tech->featureSize * LINEAR_REGION_RATIO / (resTg * 2);
  widthTgP = CalculateOnResistance(tech->featureSize, PMOS, param->temp, tech) *
             tech->featureSize * LINEAR_REGION_RATIO / (resTg * 2);
}

void Mux::CalculateArea(double _newHeight, double _newWidth,
                        AreaModify _option) {
  // cout << "********** Mux::CalculateArea Begins **********" << endl;
  double hTg, wTg;
  int numTg = numOfMux * numOfInput;
  if (_newWidth && _option == NONE) {
    double minCellWidth = 2 * (POLY_WIDTH + MIN_GAP_BET_GATE_POLY) *
                          tech->featureSize; // min standard cell width
    if (minCellWidth > _newWidth) {
      throw runtime_error("(Mux.cpp)[Error]: Mux width is even larger than the "
                          "assigned width !!!");
    }
    int numOfTgPerRow = _newWidth / minCellWidth;
    int numOfRowsTg = ceil((double)numTg / numOfTgPerRow);
    numOfTgPerRow = ceil((double)numTg / numOfRowsTg);
    double TgWidth = _newWidth / numOfTgPerRow;
    int numFold = TgWidth / (0.5 * minCellWidth) - 1;
    // widthTgN, widthTgP and numFold can determine the height and width of each
    // pass gate
    CalculatePassGateArea(widthTgN, widthTgP, tech, numFold, &hTg, &wTg);
    width = _newWidth;
    height = hTg * numOfRowsTg;
    // cout << "width = " << width << endl;
    // cout << "minCellWidth = " << minCellWidth << endl;
    // cout << "TgWidth = " << TgWidth << endl;
    // cout << "numOfTgPerRow = " << numOfTgPerRow << endl;
    // cout << "numOfRowsTg = " << numOfRowsTg << endl;
    // cout << "numFold = " << numFold << endl;
    // cout << "hTg = " << hTg << endl;
    // cout << "widthTgN = " << widthTgN << endl;
    // cout << "widthTgP = " << widthTgP << endl;
    // cout << "height = " << height << endl;
    // cout << "width = " << width << endl;
    // cout << "area = " << width * height << endl;
    // cout << "resTg = " << resTg << endl;
  } else {
    // no Folding
    width = wTg * numTg;
    height = hTg;
  }
  area = height * width;
  
  if (_option == OVERRIDE) {
    width = _newWidth;
    height = _newHeight;
    area = height * width;
  }

  capTgGateN = CalculateGateCap(widthTgN, tech);
  capTgGateP = CalculateGateCap(widthTgP, tech);
  CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech, nullptr,
                           &capTgDrain);
  // cout << "hTg = " << hTg << endl;
  // cout << "********** Mux::CalculateArea Ends **********" << endl;
}

void Mux::CalculateLatency(double _capLoad, double numRead) {
  double resPullUp;
  double tauf; // Time constant
  // Calibration: use resTg*2 (only one transistor is transmitting signal in the
  // pass gate) may be more accurate, and include gate cap because the voltage
  // at the source of NMOS and drain of PMOS is changing (Cg = 0.5Cgs + 0.5Cgd)
  tauf =
      resTg * 2 * (capTgDrain + 0.5 * capTgGateN + 0.5 * capTgGateP + _capLoad);
  readLatency = 2.3 * tauf * numRead;
}

void Mux::CalculatePower(double numRead) {
  // No Leakage in Mux
  // Selected pass gates (OFF to ON)
  // cout << "********** Mux::CalculatePower Begins **********" << endl;
  readDynamicEnergy += capTgGateN * numOfMux * tech->vdd * tech->vdd;
  readDynamicEnergy +=
      (capTgDrain * 2) * numOfMux * param->readVoltage * param->readVoltage;
  readDynamicEnergy *= numRead;
  // cout << "readDynamicEnergy = " << readDynamicEnergy << endl;
  // cout << "capTgGateN = " << capTgGateN << endl;
  // cout << "capTgDrain = " << capTgDrain << endl;
  // cout << "numInput = " << numOfMux << endl;
  // cout << "widthTgN = " << widthTgN << endl;
  // cout << "widthTgP = " << widthTgP << endl;
  // cout << "********** Mux::CalculatePower Ends **********" << endl;
}

} // namespace CoMN
