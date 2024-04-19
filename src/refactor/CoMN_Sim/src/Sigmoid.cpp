/**
 * @file Sigmoid.cpp
 * @author booniebears
 * @brief Default: Used in NVM.
 * @date 2024-01-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "../include/Sigmoid.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {

Sigmoid::Sigmoid(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit(),
      wlDecoder(_param, _technology), voltageSenseAmp(_param, _technology) {}

void Sigmoid::Initialize(int _numOfYBits, int _numOfEntries, int _numOfUnits) {
  // Y = Sigmoid(X). Sigmoid is implemented with lookup tables.
  numOfYBits = _numOfYBits;
  numOfEntries = _numOfEntries; // num of (X,Y) entry pairs.
  numOfUnits = _numOfUnits;

  cellWidth = (param->accessType == CMOS_access)
                  ? param->widthInFeatureSize1T1R
                  : param->widthInFeatureSizeCrossbar;
  cellHeight = (param->accessType == CMOS_access)
                   ? param->heightInFeatureSize1T1R
                   : param->heightInFeatureSizeCrossbar;

  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = widthInvN * tech->pnSizeRatio;
  wlDecoder.Initialize(REGULAR_ROW, ceil(log2((double)numOfEntries)), false,
                       false);
  voltageSenseAmp.Initialize(numOfYBits);
}

void Sigmoid::CalculateArea(double _newHeight, double _newWidth,
                            AreaModify _option) {
  double hUnit = cellHeight * tech->featureSize;
  double wUnit = cellWidth * tech->featureSize * numOfYBits;
  wlDecoder.CalculateArea(0, 0, NONE);
  voltageSenseAmp.CalculateArea();
  area = hUnit * wUnit * numOfEntries + wlDecoder.area + voltageSenseAmp.area;
  area *= numOfUnits;
  if (_newWidth && _option == NONE) {
    width = _newWidth;
    height = area / width;
  } else {
    height = _newHeight;
    width = area / height;
  }
}

void Sigmoid::CalculateLatency(double numRead) {
  double widthAccessCMOS;
  if (param->memcellType == Type::SRAM) {
    widthAccessCMOS = param->widthAccessCMOS;
  } else {
    double resCellAccess = param->resistanceOn * IR_DROP_TOLERANCE;
    widthAccessCMOS =
        CalculateOnResistance(tech->featureSize, NMOS, param->temp, tech) *
        LINEAR_REGION_RATIO / resCellAccess; // get access CMOS width
  }
  double capCellAccess =
      CalculateDrainCap(widthAccessCMOS * tech->featureSize, NMOS,
                        cellWidth * tech->featureSize, tech);
  capSRAMCell = capCellAccess +
                CalculateDrainCap(param->widthSRAMCellNMOS * tech->featureSize,
                                  NMOS, cellWidth * tech->featureSize, tech) +
                CalculateDrainCap(param->widthSRAMCellPMOS * tech->featureSize,
                                  PMOS, cellWidth * tech->featureSize, tech);
  wlDecoder.CalculateLatency(1e20, 0, capCellAccess, 1, 1);
  voltageSenseAmp.CalculateLatency(1);
  readLatency = wlDecoder.readLatency + voltageSenseAmp.readLatency;
  readLatency *= numRead;
}

void Sigmoid::CalculatePower(double numRead) {
  wlDecoder.CalculatePower(1, 1);
  voltageSenseAmp.CalculatePower(1);
  leakage = voltageSenseAmp.leakage + wlDecoder.leakage;
  readDynamicEnergy =
      voltageSenseAmp.readDynamicEnergy + wlDecoder.readDynamicEnergy;
  readDynamicEnergy *= numRead * numOfUnits;
}

} // namespace CoMN
