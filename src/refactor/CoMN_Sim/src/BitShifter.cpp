/**
 * @file BitShifter.cpp
 * @author booniebears
 * @brief The implementation of ReLU unit.
 * @date 2023-11-07
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "../include/BitShifter.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
BitShifter::BitShifter(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit(), dff(_param, _technology) {}

void BitShifter::Initialize(int _numOfBits, int _numOfUnits) {
  numOfBits = _numOfBits;
  numOfUnits = _numOfUnits;
  numOfDFFs = numOfUnits * numOfBits;
  dff.Initialize(numOfDFFs);
}

void BitShifter::CalculateArea(double _newHeight, double _newWidth,
                               AreaModify _option) {
  dff.CalculateArea(0, 0, NONE); // get one row of DFFs by default
  area = dff.area;
  if (_newWidth && _option == NONE) {
    width = _newWidth;
    height = area / width;
  } else {
    height = _newHeight;
    width = area / height;
  }
}

void BitShifter::CalculateLatency(double numRead) {
  dff.CalculateLatency(1);
  readLatency = dff.readLatency * numRead;
}

void BitShifter::CalculatePower(double numRead) {
  // cout << "********** BitShifter::CalculatePower Begins **********" << endl;
  dff.CalculatePower(numRead, numOfDFFs);
  leakage += dff.leakage;
  readDynamicEnergy += dff.readDynamicEnergy;
  // cout << "numRead = " << numRead << endl;
  // cout << "numOfDFFs = " << numOfDFFs << endl;
  // cout << "numOfBits = " << numOfBits << endl;
  // cout << "numOfUnits = " << numOfUnits << endl;
  // cout << "********** BitShifter::CalculatePower Ends **********" << endl;
}

} // namespace CoMN
