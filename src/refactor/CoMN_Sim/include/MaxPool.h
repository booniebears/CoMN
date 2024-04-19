/**
 * @file MaxPool.h
 * @author booniebears
 * @brief
 * @date 2023-12-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MAXPOOL_H_
#define MAXPOOL_H_

#include "../include/BasicUnit.h"
#include "../include/Comparator.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {

class MaxPool : public BasicUnit {
public:
  MaxPool(Param *_param, Technology *_technology);
  virtual ~MaxPool() {}
  void Initialize(int _numOfBits, int _window, int _numOfUnits);
  void CalculateArea(double widthArray);
  void CalculateLatency(double _rampInput, double _capLoad, double numRead);
  void CalculatePower(double numRead);

  int numOfBits;
  int numOfUnits;

private:
  Param *param;
  Technology *tech;
  Comparator comparator;
  int numOfComps = 0;     // num of Comparators
  int numOfComStages = 0; // num of Compare stages

  double widthInvN, widthInvP, widthNandN, widthNandP, widthNorN, widthNorP,
      widthNorN2, widthNorP2;

  double capInvInput, capInvOutput, capNandInput, capNandOutput, capNorInput,
      capNorOutput, capNor2Input, capNor2Output;
};

} // namespace CoMN

#endif // !MAXPOOL_H_