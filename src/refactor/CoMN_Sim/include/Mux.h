#ifndef MUX_H_
#define MUX_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class Mux : public BasicUnit {
public:
  Mux(Param *_param, Technology *_technology);
  virtual ~Mux() {}
  void Initialize(int _numOfMux, int _numOfInput, double _resTg);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _capLoad, double numRead);
  void CalculatePower(double numRead);

  int numOfInput;
  int numOfMux; // How many mux units
  double resTg;
  double capTgGateN, capTgGateP, capTgDrain;

private:
  Param *param;
  Technology *tech;
  // Transistor Parameters
  double widthTgN, widthTgP;
};
} // namespace CoMN

#endif