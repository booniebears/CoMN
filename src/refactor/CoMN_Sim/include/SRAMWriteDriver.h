/**
 * @file SRAMWriteDriver.h
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef SRAMWRITEDRIVER_H_
#define SRAMWRITEDRIVER_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class SRAMWriteDriver : public BasicUnit {
public:
  SRAMWriteDriver(Param *_param, Technology *_technology);
  virtual ~SRAMWriteDriver() {}
  void Initialize(int _numOfLines, double _activityColWrite);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double _capLoad, double _resLoad,
                        double numWrite);
  void CalculatePower(double numWrite);

  int numOfLines;
  double activityColWrite;

private:
  Param *param;
  Technology *tech;
  double widthInvN, widthInvP;
  double capInvInput, capInvOutput;
};
} // namespace CoMN

#endif