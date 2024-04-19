/**
 * @file ShiftAdder.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <cmath>

#include "../include/ShiftAdder.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
ShiftAdder::ShiftAdder(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit(), adder(_param, _technology),
      dff(_param, _technology) {}

void ShiftAdder::Initialize(int _numOfBits, int _numOfAdders,
                            int _numOfReadPulses, SpikingMode _spikingMode) {
  numOfBits = _numOfBits;
  numOfAdders = _numOfAdders;
  numOfReadPulses = _numOfReadPulses;
  spikingMode = _spikingMode;
  if (spikingMode == NONSPIKING) { // Need adder
    adder.Initialize(numOfBits, numOfAdders);
    // numOfBits + 1 + numOfReadPulses - 1
    numOfDFFs = (numOfBits + numOfReadPulses) * numOfAdders;
    dff.Initialize(numOfDFFs);
  } else { // No adder, just count
    numOfDFFs = numOfAdders * pow(2, numOfBits);
    dff.Initialize(numOfDFFs);
  }
  /* Currently ignore INV and NAND in shift-add circuit */
  // PISO shift register (https://en.wikipedia.org/wiki/Shift_register)
  // INV
  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
  numOfInvs = numOfAdders;
  // NAND2
  widthNandN = 2 * MIN_NMOS_SIZE * tech->featureSize;
  widthNandP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
  // numOfDFFs / numOfAdders means the number of DFF for each shift adder
  numOfNands = 3 * (numOfDFFs / numOfAdders - 1) * numOfAdders;
}

void ShiftAdder::CalculateArea(double _newHeight, double _newWidth,
                               AreaModify _option) {
  //
  double hInv, wInv, hNand, wNand;
  // INV
  CalculateGateArea(INV, 1, widthInvN, widthInvP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv,
                    &wInv);
  // NAND2
  CalculateGateArea(NAND, 2, widthNandN, widthNandP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand,
                    &wNand);
  if (_newWidth && _option == NONE) {
    if (spikingMode == NONSPIKING) {
      adder.CalculateArea(0, _newWidth, NONE);
      dff.CalculateArea(0, _newWidth, NONE);
      width = _newWidth;
      height =
          adder.height + tech->featureSize * MAX_TRANSISTOR_HEIGHT + dff.height;
    } else {
      dff.CalculateArea(0, _newWidth, NONE);
      width = _newWidth;
      height = tech->featureSize * MAX_TRANSISTOR_HEIGHT + dff.height;
    }
  } else {
    if (spikingMode == NONSPIKING) {
      adder.CalculateArea(0, _newWidth, NONE);
      dff.CalculateArea(0, _newWidth, NONE);
      width = adder.width + wInv + wNand + dff.width;
      height = _newHeight;
    } else {
      dff.CalculateArea(0, _newWidth, NONE);
      width = wInv + wNand + dff.width;
      height = _newHeight;
    }
  }
  area = width * height;
}

void ShiftAdder::CalculateLatency(double numRead) {
  // Assume the delay of INV and NAND2 are negligible
  if (spikingMode == NONSPIKING) {
    adder.CalculateLatency(INF_RAMP, dff.getCap(), 1);
    dff.CalculateLatency(1);
    double shiftAddLatency = adder.readLatency + dff.readLatency;
    // Consider the frequency of read pulses during several reads;
    if (shiftAddLatency > param->readPulseWidth)
      readLatency += (shiftAddLatency - param->readPulseWidth) * (numRead - 1);
    readLatency += shiftAddLatency;
  } else {
    dff.CalculateLatency(1);
    double shiftLatency = dff.readLatency;
    if (shiftLatency > param->readPulseWidth)
      readLatency += (shiftLatency - param->readPulseWidth) * (numRead - 1);
    readLatency += shiftLatency;
  }
}

void ShiftAdder::CalculatePower(double numRead) {
  if (spikingMode == NONSPIKING) {
    adder.CalculatePower(numRead, numOfAdders);
    dff.CalculatePower(numRead, numOfDFFs);
    leakage = adder.leakage + dff.leakage;
    readDynamicEnergy = adder.readDynamicEnergy + dff.readDynamicEnergy;
  } else {
    dff.CalculatePower(numRead, numOfDFFs);
    leakage = dff.leakage;
    readDynamicEnergy = adder.readDynamicEnergy;
  }
}

} // namespace CoMN
