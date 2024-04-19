/**
 * @file VoltageSenseAmp.h
 * @author booniebears
 * @brief Voltage Sense amplifier for Sigmoid Implementation.
 * @date 2024-01-19
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef VOLTAGESENSEAMP_H_
#define VOLTAGESENSEAMP_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {

class VoltageSenseAmp : public BasicUnit {

public:
  VoltageSenseAmp(Param *_param, Technology *_technology);
  virtual ~VoltageSenseAmp() {}

  void Initialize(int _numOfCols);
  void CalculateArea();
  void CalculateLatency(double numRead);
  void CalculatePower(double numRead);

private:
  /* data */
  Param *param;
  Technology *tech;

  int numOfCols;
  double widthNMOS, widthPMOS;
  double widthVoltageSenseAmp;
  double voltageSenseDiff;
  double resPrecharge;
  double capNMOSGate, capNMOSDrain, capPMOSGate, capPMOSDrain;
  double capS1;
};

} // namespace CoMN

#endif // !VOLTAGESENSEAMP_H_