/**
 * @file DFF.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/DFF.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
DFF::DFF(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void DFF::Initialize(int _numDFF) {
  numOfDFFs = _numDFF;
  widthTgN = MIN_NMOS_SIZE * tech->featureSize;
  widthTgP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
}

void DFF::CalculateArea(double _newHeight, double _newWidth,
                        AreaModify _option) {
  double hDFFInv, wDFFInv, hDFF, wDFF;
  CalculateGateArea(INV, 1, MIN_NMOS_SIZE * tech->featureSize,
                    tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hDFFInv,
                    &wDFFInv);
  hDFF = hDFFInv;
  wDFF = wDFFInv * 12; // DFF = 12 cells put together
  if (_newWidth && _option == NONE) {
    int numOfDFFsPerRow = _newWidth / wDFF;
    if (numOfDFFsPerRow > numOfDFFs)
      numOfDFFsPerRow = numOfDFFs;
    int numOfRowsDFF = ceil((double)numOfDFFs / numOfDFFsPerRow);
    width = _newWidth;
    height = hDFF * numOfRowsDFF;
  } else if (_newHeight && _option == NONE) {
    int numOfDFFsPerCol = _newHeight / hDFF;
    if (numOfDFFsPerCol > numOfDFFs)
      numOfDFFsPerCol = numOfDFFs;
    int numOfColumnsDFF = ceil((double)numOfDFFs / numOfDFFsPerCol);
    width = wDFF * numOfColumnsDFF;
    height = _newHeight;
  } else { // One row of DFFs by default
    width = wDFF * numOfDFFs;
    height = hDFF;
  }
  area = width * height;

  // Capacitance
  // INV
  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hDFFInv, tech,
                           &capInvInput, &capInvOutput);
  // TG
  capTgGateN = CalculateGateCap(widthTgN, tech);
  capTgGateP = CalculateGateCap(widthTgP, tech);
  CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hDFFInv, tech, NULL,
                           &capTgDrain);
}

void DFF::CalculateLatency(double numRead) {
  readLatency = (1.0 / 2 / param->clkFreq) * numRead;
}

void DFF::CalculatePower(double numRead, int numDffPerOperation) {
  leakage =
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
      tech->vdd * 8 * numOfDFFs;

  // CLK INV (all DFFs have energy consumption)
  readDynamicEnergy +=
      (capInvInput + capInvOutput) * tech->vdd * tech->vdd * 4 * numOfDFFs;
  // CLK TG (all DFFs have energy consumption)
  readDynamicEnergy += capTgGateN * tech->vdd * tech->vdd * 2 * numOfDFFs;
  readDynamicEnergy += capTgGateP * tech->vdd * tech->vdd * 2 * numOfDFFs;
  // D to Q path (only selected DFFs have energy consumption)
  readDynamicEnergy += (capTgDrain * 3 + capInvInput) * tech->vdd * tech->vdd *
                       min(numDffPerOperation, numOfDFFs); // D input side
  readDynamicEnergy += (capTgDrain + capInvOutput) * tech->vdd * tech->vdd *
                       min(numDffPerOperation, numOfDFFs); // D feedback side
  readDynamicEnergy += (capInvInput + capInvOutput) * tech->vdd * tech->vdd *
                       min(numDffPerOperation, numOfDFFs); // Q output side

  readDynamicEnergy *= numRead;
}

} // namespace CoMN
