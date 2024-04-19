/**
 * @file DFF.h
 * @author booniebears
 * @brief
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef DFF_H_
#define DFF_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class DFF : public BasicUnit {
public:
  DFF(Param *_param, Technology *_technology);
  virtual ~DFF() {}
  void Initialize(int _numDFF);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double numRead);
  void CalculatePower(double numRead, int numDffPerOperation);
  double getCap() { return capTgDrain; }
  int numOfDFFs;

private:
  Param *param;
  Technology *tech;

  // Transistor attributes
  double widthTgN, widthTgP, widthInvN, widthInvP;
  double capTgDrain, capTgGateN, capTgGateP, capInvInput, capInvOutput;
};
} // namespace CoMN

#endif