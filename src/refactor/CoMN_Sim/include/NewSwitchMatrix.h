/**
 * @file NewSwitchMatrix.h
 * @author booniebears
 * @brief
 * @date 2023-10-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef NEWSWITCHMATRIX_H_
#define NEWSWITCHMATRIX_H_

#include "../include/BasicUnit.h"
#include "../include/DFF.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class NewSwitchMatrix : public BasicUnit {
public:
  NewSwitchMatrix(Param *_param, Technology *_technology);
  virtual ~NewSwitchMatrix() {}
  void Initialize(int _numOfLines, double _activityRowRead);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double _capLoad, double _resLoad,
                        double numRead, double numWrite);
  void CalculatePower(double numRead, double numWrite);

  int numOfLines; // Num of Output Lines connected to SwitchMatrix
  double activityRowRead;

  DFF dff;

private:
  Param *param;
  Technology *tech;
  double widthTgN, widthTgP;
  double resTg;
  double capTgDrain, capTgGateN, capTgGateP;
};
} // namespace CoMN

#endif