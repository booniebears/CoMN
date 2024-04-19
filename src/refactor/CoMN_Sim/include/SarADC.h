/**
 * @file SarADC.h
 * @author booniebears
 * @brief
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef SARADC_H_
#define SARADC_H_

#include <vector>

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class SarADC : public BasicUnit {
public:
  SarADC(Param *_param, Technology *_technology);
  virtual ~SarADC() {}
  void Initialize(int _levelOutput, int _numOfADCs);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double numRead);
  void CalculatePower(vector<double> &colResistance, double numRead);
  double GetColumnPower(double resCol);

  int levelOutput;
  int numOfADCs;

private:
  Param *param;
  Technology *tech;
  double widthNMOS, widthPMOS;
};
} // namespace CoMN

#endif