/**
 * @file Precharger.h
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef PRECHARGER_H_
#define PRECHARGER_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class Precharger : public BasicUnit {
public:
  Precharger(Param *_param, Technology *_technology);
  virtual ~Precharger() {}
  void Initialize(int _numOfLines, double _resLoad, double _activityColWrite);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _capLoad, double numRead, double numWrite);
  void CalculatePower(double numRead, double numWrite);

  int numOfLines;
  double resLoad;
  double activityColWrite;

private:
  Param *param;
  Technology *tech;
  double widthPMOSBitlinePrecharger, widthPMOSBitlineEqual;
  double capOutputBitlinePrecharger;
  double capLoad;
};
} // namespace CoMN

#endif