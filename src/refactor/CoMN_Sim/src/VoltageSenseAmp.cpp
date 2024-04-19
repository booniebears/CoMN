/**
 * @file VoltageSenseAmp.cpp
 * @author booniebears
 * @brief Voltage Sense amplifier for Sigmoid Implementation.
 * @date 2024-01-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "../include/VoltageSenseAmp.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {

VoltageSenseAmp::VoltageSenseAmp(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void VoltageSenseAmp::Initialize(int _numOfCols) {
  // The SenseAmp Units are aligned in a row.
  numOfCols = _numOfCols;
  widthNMOS = MIN_NMOS_SIZE * tech->featureSize;
  widthPMOS = widthNMOS * tech->pnSizeRatio;
  voltageSenseDiff = 0.1;
}

void VoltageSenseAmp::CalculateArea() {
  // width/height not calculated here.
  double hNMOS, wNMOS, hPMOS, wPMOS;

  CalculateGateArea(INV, 1, widthNMOS, 0,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNMOS,
                    &wNMOS);
  CalculateGateArea(INV, 1, 0, widthPMOS,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hPMOS,
                    &wPMOS);
  double unitArea = (hNMOS * wNMOS) * 6 + (hPMOS * wPMOS) * 5;

  area = unitArea * numOfCols;
  // Resistance
  resPrecharge = CalculateOnResistance(widthPMOS, PMOS, param->temp, tech);
  // Capacitance
  CalculateGateCapacitance(INV, 1, widthNMOS, 0, hNMOS, tech, &capNMOSGate,
                           &capNMOSDrain);
  CalculateGateCapacitance(INV, 1, 0, widthPMOS, hPMOS, tech, &capPMOSGate,
                           &capPMOSDrain);
  capS1 = capNMOSGate + capNMOSDrain + capPMOSDrain;
}

void VoltageSenseAmp::CalculateLatency(double numRead) {
  // capInputLoad = 0 in sigmoid.
  double resMemCellOn;
  double resMemCellOff;
  if (param->accessType == CMOS_access) {
    resMemCellOn = param->resistanceOn * (1 + IR_DROP_TOLERANCE);
    resMemCellOff = param->resistanceOff * (1 + IR_DROP_TOLERANCE);
  } else {
    resMemCellOn = param->resistanceOn;
    resMemCellOff = param->resistanceOff;
  }

  readLatency +=
      2.3 * resPrecharge * capS1 + voltageSenseDiff * (capS1 + capNMOSDrain) /
                                       (param->readVoltage / resMemCellOn -
                                        param->readVoltage / resMemCellOff);
  // Clock time for precharge and S/A enable
  readLatency += 1 / param->clkFreq * 2;
  readLatency *= numRead;
}

void VoltageSenseAmp::CalculatePower(double numRead) {
  // Leakage (assume connection to the cell is floating, no leakage on the
  // precharge side, but in S/A it's roughly like 2 NAND2)
  leakage +=
      CalculateGateLeakage(NAND, 2, widthNMOS, widthPMOS, param->temp, tech) *
      tech->vdd * 2;

  readDynamicEnergy =
      9.845e-15 * (tech->vdd / 1.1) * (tech->vdd / 1.1); // 65nm tech node
  readDynamicEnergy *= numOfCols;
  readDynamicEnergy *= numRead;
}

} // namespace CoMN
