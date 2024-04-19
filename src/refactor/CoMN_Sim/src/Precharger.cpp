/**
 * @file Precharger.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/Precharger.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
Precharger::Precharger(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void Precharger::Initialize(int _numOfLines, double _resLoad,
                            double _activityColWrite) {
  numOfLines = _numOfLines;
  resLoad = _resLoad;
  activityColWrite = _activityColWrite;

  widthPMOSBitlineEqual = MIN_NMOS_SIZE * tech->featureSize;
  widthPMOSBitlinePrecharger = 6 * tech->featureSize;
}

void Precharger::CalculateArea(double _newHeight, double _newWidth,
                               AreaModify _option) {
  double hBitlinePrecharger, wBitlinePrecharger;
  double hBitlineEqual, wBitlineEqual; // TODO: what?
  CalculateGateArea(INV, 1, 0, widthPMOSBitlinePrecharger,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech,
                    &hBitlinePrecharger, &wBitlinePrecharger);
  CalculateGateArea(INV, 1, 0, widthPMOSBitlineEqual,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech,
                    &hBitlineEqual, &wBitlineEqual);
  double hUnit = hBitlinePrecharger + hBitlineEqual * 2;
  double wUnit = max(wBitlinePrecharger, wBitlineEqual);
  if (_newWidth && _option == NONE) {
    width = _newWidth;
    int numOfUnitsPerLine = _newWidth / wUnit;
    if (numOfUnitsPerLine > numOfLines)
      numOfUnitsPerLine = numOfLines;
    // Per unit occupies ? lines
    int numOfLinesUnit = ceil((double)numOfLines / numOfUnitsPerLine);
    width = _newWidth;
    height = numOfLinesUnit * hUnit;
  } else {
    width = numOfLines * wUnit;
    height = hUnit;
  }
  area = width * height;

  // Capacitance
  capOutputBitlinePrecharger =
      CalculateDrainCap(widthPMOSBitlinePrecharger, PMOS, hBitlinePrecharger,
                        tech) +
      CalculateDrainCap(widthPMOSBitlineEqual, PMOS, hBitlineEqual, tech);
}

void Precharger::CalculateLatency(double _capLoad, double numRead,
                                  double numWrite) {
  double gm;   /* transconductance */
  double beta; /* for horowitz calculation */
  double resPullUp;
  double tauf;
  capLoad = _capLoad; // Save capLoad here, and use it in CalculatePower.

  resPullUp = CalculateOnResistance(widthPMOSBitlinePrecharger, PMOS,
                                    param->temp, tech);
  gm = CalculateTransconductance(widthPMOSBitlinePrecharger, PMOS, tech);
  beta = 1 / (gm * resPullUp);
  tauf = resPullUp * (_capLoad + capOutputBitlinePrecharger) +
         resLoad * _capLoad / 2; // τf = ΣRC in the horowitz func
  readLatency = horowitz(tauf, beta, INF_RAMP, nullptr);
  writeLatency = readLatency;
  readLatency *= numRead;
  writeLatency *= numWrite;
}

void Precharger::CalculatePower(double numRead, double numWrite) {
  leakage = CalculateGateLeakage(INV, 1, 0, widthPMOSBitlinePrecharger,
                                 param->temp, tech) *
            tech->vdd * numOfLines;
  // TODO: BL and BL_bar?? *2?
  readDynamicEnergy = capLoad * tech->vdd * tech->vdd * numOfLines * 2;
  readDynamicEnergy *= numRead;

  writeDynamicEnergy =
      capLoad * tech->vdd * tech->vdd * numOfLines * activityColWrite;
  writeDynamicEnergy *= numWrite;
}

} // namespace CoMN
