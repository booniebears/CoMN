/**
 * @file Comparator.h
 * @author booniebears
 * @brief Pairwise Comparator.
 * @date 2023-12-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef COMPARATOR_H_
#define COMPARATOR_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {

class Comparator : public BasicUnit {
public:
  Comparator(Param *_param, Technology *_technology);
  virtual ~Comparator() {}
  void Initialize(int _numOfBits, int _numOfUnits);
  void CalculateArea(double widthArray);
  void CalculateLatency(double _rampInput, double _capLoad, double numRead);
  void CalculatePower(double numRead);

  int numOfBits;
  int numOfUnits;
  double areaUnit;

private:
  Param *param;
  Technology *tech;
  double widthInvN, widthInvP, widthNand2N, widthNand2P, widthNand3N,
      widthNand3P;
  double capInvInput, capInvOutput, capNand2Input, capNand2Output,
      capNand3Input, capNand3Output;
};

} // namespace CoMN

#endif // !COMPARATOR_H_
