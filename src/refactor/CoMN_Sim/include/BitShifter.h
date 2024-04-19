/**
 * @file BitShifter.h
 * @author booniebears
 * @brief The implementation of ReLU unit.
 * @date 2023-11-07
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef BITSHIFTER_H_
#define BITSHIFTER_H_

#include "../include/BasicUnit.h"
#include "../include/DFF.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class BitShifter : public BasicUnit {
public:
  BitShifter(Param *_param, Technology *_technology);
  virtual ~BitShifter() {}
  void Initialize(int _numOfBits, int _numOfUnits);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double numRead);
  void CalculatePower(double numRead);

  int numOfBits;
  int numOfUnits;

private:
  Param *param;
  Technology *tech;
  DFF dff;

  int numOfDFFs;
};
} // namespace CoMN

#endif // BITSHIFTER_H_