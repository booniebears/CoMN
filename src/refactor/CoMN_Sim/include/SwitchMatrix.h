/**
 * @file SwitchMatrix.h
 * @author booniebears
 * @brief
 * @date 2023-10-18
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef SWITCHMATRIX_H_
#define SWITCHMATRIX_H_

#include "../include/BasicUnit.h"
#include "../include/DFF.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class SwitchMatrix : public BasicUnit {
public:
  SwitchMatrix(Param *_param, Technology *_technology);
  virtual ~SwitchMatrix() {}
  void Initialize(RowColMode _mode, int _numOfLines, double _resTg,
                  double _activityRowRead, double _activityColWrite,
                  double _numOfWritePulses);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double _capLoad, double _resLoad,
                        double numRead, double numWrite);
  void CalculatePower(double numRead, double numWrite);
  double getRampOutput() { return rampOutput; }

  RowColMode mode;
  int numOfLines; // Num of Output Lines connected to SwitchMatrix
  double resTg;
  double activityRowRead;
  double activityColWrite; // "write" related members not necessary in training
  double numOfWritePulses; // Same as above

  DFF dff;

private:
  Param *param;
  Technology *tech;
  // Transistor Parameters
  double widthTgN, widthTgP;
  double TgHeight, TgWidth;
  double capTgDrain, capTgGateN, capTgGateP;
  double rampOutput;
};
} // namespace CoMN

#endif