/**
 * @file MultilevelSenseAmp.h
 * @author booniebears
 * @brief
 * @date 2023-10-18
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef MULTILEVELSENSEAMP_H_
#define MULTILEVELSENSEAMP_H_

#include <vector>

#include "../include/BasicUnit.h"
#include "../include/CurrentSenseAmp.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class MultilevelSenseAmp : public BasicUnit {
public:
  MultilevelSenseAmp(Param *_param, Technology *_technology);
  virtual ~MultilevelSenseAmp() {}
  void Initialize(int _levelOutput, int _numOfAmps, bool _isCSA,
                  bool _isParallel);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(vector<double> &colResistance, double numRead);
  void CalculatePower(vector<double> &colResistance, double numRead);
  double GetColumnLatency(double resCol);
  double GetColumnPower(double resCol);

  int levelOutput;
  int numOfAmps;
  bool isCSA; // use CSA or VSA?
  bool isParallel;

private:
  Param *param;
  Technology *tech;
  vector<double> Rref;
  double widthNMOS, widthPMOS;
};
} // namespace CoMN

#endif