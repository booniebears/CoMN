/**
 * @file CurrentSenseAmp.h
 * @author booniebears
 * @brief
 * @date 2023-10-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef CURRENTSENSEAMP_H_
#define CURRENTSENSEAMP_H_

#include <vector>

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class CurrentSenseAmp : public BasicUnit {
public:
  CurrentSenseAmp(Param *_param, Technology *_technology);
  virtual ~CurrentSenseAmp() {}
  void Initialize(int _numOfColumns);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(vector<double> &colResistance, double numRead);
  void CalculatePower(vector<double> &colResistance, double numRead);
  double GetColumnLatency(double resCol);
  double GetColumnPower(double resCol);

  int numOfColumns;

private:
  Param *param;
  Technology *tech;
  double widthNMOS, widthPMOS;
  double Rref;
};
} // namespace CoMN

#endif