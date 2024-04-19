#ifndef MULTILEVELSAENCODER_H_
#define MULTILEVELSAENCODER_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class MultilevelSAEncoder : public BasicUnit {
public:
  MultilevelSAEncoder(Param *_param, Technology *_technology);
  virtual ~MultilevelSAEncoder() {}
  void Initialize(int _levelOutput, int _numOfEncoders);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double numRead);
  void CalculatePower(double numRead);

  int levelOutput;
  int numOfEncoders;

private:
  Param *param;
  Technology *tech;
  int numInput, numGate;
  double widthInvN, widthInvP, widthNandN, widthNandP;
  double capNandInput, capNandOutput, capNandLgInput, capNandLgOutput,
      capInvInput, capInvOutput;
};
} // namespace CoMN

#endif