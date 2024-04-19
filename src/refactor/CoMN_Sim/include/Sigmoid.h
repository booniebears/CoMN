/**
 * @file Sigmoid.h
 * @author booniebears
 * @brief
 * @date 2024-01-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef SIGMOID_H_
#define SIGMOID_H_

#include "../include/BasicUnit.h"
#include "../include/MuxDecoder.h"
#include "../include/VoltageSenseAmp.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {

class Sigmoid : public BasicUnit {

public:
  Sigmoid(Param *_param, Technology *_technology);
  void Initialize(int _numOfYBits, int _numOfEntries, int _numOfUnits);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double numRead);
  void CalculatePower(double numRead);
  virtual ~Sigmoid() {}

private:
  /* data */
  MuxDecoder wlDecoder;
  VoltageSenseAmp voltageSenseAmp;
  Param *param;
  Technology *tech;

  int numOfYBits, numOfEntries, numOfUnits;
  double capSRAMCell, capLoad;
  double widthInvN, widthInvP;
  double cellWidth, cellHeight;
};

} // namespace CoMN

#endif // !SIGMOID_H_