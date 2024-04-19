/**
 * @file NewDecoderDriver.h
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef NEWDECODERDRIVER_H_
#define NEWDECODERDRIVER_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class NewDecoderDriver : public BasicUnit {
public:
  NewDecoderDriver(Param *_param, Technology *_technology);
  virtual ~NewDecoderDriver() {}
  void Initialize(int _numOfRows);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double _capLoad, double _resLoad,
                        double numRead, double numWrite);
  void CalculatePower(double numRead, double numWrite);

  int numOfRows;

private:
  Param *param;
  Technology *tech;
  // Transistor Params
  double widthNandN, widthNandP, widthInvN, widthInvP, widthTgN, widthTgP;
  double capNandInput, capNandOutput, capInvInput, capInvOutput, capTgGateN,
      capTgGateP, capTgDrain;
  double resTg;
};
} // namespace CoMN

#endif