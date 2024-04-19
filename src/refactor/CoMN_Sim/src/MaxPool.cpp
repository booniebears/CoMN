/**
 * @file MaxPool.cpp
 * @author booniebears
 * @brief
 * @date 2023-12-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "../include/MaxPool.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {

MaxPool::MaxPool(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit(),
      comparator(_param, _technology) {}

void MaxPool::Initialize(int _numOfBits, int _window, int _numOfUnits) {
  // MaxPool is composed of several Pairwise Comparators.
  // The Comparators form an adder-tree like structure.
  numOfBits = _numOfBits;
  numOfUnits = _numOfUnits;

  int cur_comps = _window;
  while (cur_comps) {
    numOfComps += cur_comps / 2;
    numOfComStages++;
    cur_comps /= 2;
  }
  if (_window % 2) {
    numOfComps++;
  }
  widthInvN = MIN_NMOS_SIZE * tech->featureSize;
  widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  widthNandN = 2 * MIN_NMOS_SIZE * tech->featureSize;
  widthNandP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  // NOR1
  widthNorN = 4 * MIN_NMOS_SIZE * tech->featureSize;
  widthNorP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  // NOR2 (numOfBits-1) inputs
  widthNorN2 = (numOfBits * 2) * MIN_NMOS_SIZE * tech->featureSize;
  widthNorP2 = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  // 1-bit Comparator
  comparator.Initialize(1, 1); // initialize single comparator
}

void MaxPool::CalculateArea(double widthArray) {
  double hInv, wInv, hNand, wNand, hNor, wNor, hNor2, wNor2;
  // INV
  CalculateGateArea(INV, 1, widthInvN, widthInvP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv,
                    &wInv);
  // NAND2
  CalculateGateArea(NAND, 2, widthNandN, widthNandP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand,
                    &wNand);
  // NOR (2 inputs)
  CalculateGateArea(NOR, 2, widthNorN, widthNorP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNor,
                    &wNor);
  // NOR (numOfBits-1 inputs)
  CalculateGateArea(NOR, numOfBits, widthNorN2, widthNorP2,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNor2,
                    &wNor2);
  // 1-bit comparator
  comparator.CalculateArea(widthArray);
  double areaUnit = ((comparator.areaUnit + (hInv * wInv) * 5) * numOfBits);
  // each N-bit comparator needs 3*TG, 3*INV, 1*AND, 1*NOR and 1*OR
  areaUnit += ((hInv * wInv) * 6) + (hNand * wNand + hInv * wInv) +
              (hNor * wNor) + (hNor2 * wNor2);
  areaUnit *= numOfComps; // each MPU need *(numOfComps) N-bit omparators

  area = areaUnit * numOfUnits;
  width = widthArray;
  height = area / width;

  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech,
                           &capInvInput, &capInvOutput);
  CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech,
                           &capNandInput, &capNandOutput);
  CalculateGateCapacitance(NOR, 2, widthNorN, widthNorP, hNor, tech,
                           &capNorInput, &capNorOutput);
  CalculateGateCapacitance(NOR, numOfBits, widthNorN2, widthNorP2, hNor2, tech,
                           &capNor2Input, &capNor2Output);
}

void MaxPool::CalculateLatency(double _rampInput, double _capLoad,
                               double numRead) {
  double tr;
  double gm;
  double beta;
  double resNOR, resINV, resTG;
  double rampNOROutput, rampINVOutput, rampTGOutput;

  comparator.CalculateLatency(_rampInput, capInvOutput * 2, 1);
  // Average numOfBits/2 bits are compared.
  readLatency += comparator.readLatency * numOfBits / 2;

  // NOR2
  resNOR = CalculateOnResistance(widthNorN2, NMOS, param->temp, tech) * 2;
  tr = resNOR * (capInvInput * 2 + numOfBits * capInvOutput);
  gm = CalculateTransconductance(widthNorN2, NMOS, tech);
  beta = 1 / (resNOR * gm);
  readLatency += horowitz(tr, beta, 1e20, &rampNOROutput);
  // INV
  resINV = CalculateOnResistance(widthInvN, NMOS, param->temp, tech) * 2;
  tr = resINV * (capInvInput + capNor2Output);
  gm = CalculateTransconductance(widthInvN, NMOS, tech);
  beta = 1 / (resINV * gm);
  readLatency += horowitz(tr, beta, 1e20, &rampINVOutput);
  // TG
  resTG = CalculateOnResistance(widthInvN, NMOS, param->temp, tech) * 2;
  tr = resTG * (capInvInput * 2 + _capLoad);
  gm = CalculateTransconductance(widthInvN, NMOS, tech);
  beta = 1 / (resTG * gm);
  readLatency += horowitz(tr, beta, rampINVOutput, &rampTGOutput);

  readLatency *= numOfComStages;
  readLatency *= numRead;
}

void MaxPool::CalculatePower(double numRead) {
  leakage +=
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
      tech->vdd * numOfComps * numOfUnits;
  comparator.CalculatePower(1); // single 1-bit comparator read only once

  readDynamicEnergy += (capNandInput + capInvOutput) * tech->vdd * tech->vdd;
  readDynamicEnergy += (capNorInput + capInvOutput) * tech->vdd * tech->vdd;
  readDynamicEnergy += (capInvInput + capInvOutput) * tech->vdd * tech->vdd * 2;
  readDynamicEnergy += (capNor2Input + capInvOutput) * tech->vdd * tech->vdd;
  // Average numOfBits/2 bits are compared.
  readDynamicEnergy += comparator.readDynamicEnergy * numOfBits / 2;

  readDynamicEnergy *= numOfComps; // need *numComparator N-bit comparator
  readDynamicEnergy *= numOfUnits;
  readDynamicEnergy *= numRead;
}

} // namespace CoMN
