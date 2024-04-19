/**
 * @file SenseAmp.h
 * @author booniebears
 * @brief Sense Amplifier for readout of subarray columns.
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef SENSEAMP_H_
#define SENSEAMP_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class SenseAmp : public BasicUnit {
public:
  SenseAmp(Param *_param, Technology *_technology);
  virtual ~SenseAmp() {}
  void Initialize(int _numOfColumns, bool _isCurrentSense, double _senseVoltage,
                 double _pitchSenseAmp);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double numRead);
  void CalculatePower(double numRead);

  int numOfColumns;
  bool isCurrentSense;  // Current sense based?
  double senseVoltage;  // Minimum sensible voltage
  double pitchSenseAmp; // maximum width allowed for 1 sense amplifier layout

private:
  Param *param;
  Technology *tech;
  double capLoad; /* Load capacitance of sense amplifier */
};
} // namespace CoMN

#endif