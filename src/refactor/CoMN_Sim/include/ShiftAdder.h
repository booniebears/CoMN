#ifndef SHIFTADDER_H_
#define SHIFTADDER_H_

#include "../include/Adder.h"
#include "../include/BasicUnit.h"
#include "../include/DFF.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class ShiftAdder : public BasicUnit {
public:
  ShiftAdder(Param *_param, Technology *_technology);
  virtual ~ShiftAdder() {}
  void Initialize(int _numOfBits, int _numOfAdders, int _numOfReadPulses,
                  SpikingMode _spikingMode);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double numRead);
  void CalculatePower(double numRead);

  int numOfAdders;
  int numOfBits;
  int numOfReadPulses;
  int numOfDFFs;
  SpikingMode spikingMode; // Binary: Adder; Spiking: Counting

  // Transistor attributes
  double widthInvN, widthInvP, widthNandN, widthNandP;
  int numOfInvs, numOfNands;

  Adder adder;
  DFF dff;

private:
  Param *param;
  Technology *tech;
};
} // namespace CoMN

#endif