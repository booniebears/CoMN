/**
 * @file Adder.h
 * @author booniebears
 * @brief
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef ADDER_H_
#define ADDER_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class Adder : public BasicUnit {
public:
  Adder(Param *_param, Technology *_technology);
  virtual ~Adder() {}
  void Initialize(int _numOfBits, int _numOfAdders);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double _capLoad, double numRead);
  void CalculatePower(double numRead, int numAdderPerOperation);

  int numOfAdders; // record that so as to Calculate Area/Latency Later;
  int numOfBits;   // num of bits of an Adder

private:
  Param *param;
  Technology *tech;
  double widthNandN, widthNandP; // NMOS and PMOS of NAND
  double capNandInput, capNandOutput;
};
} // namespace CoMN

#endif