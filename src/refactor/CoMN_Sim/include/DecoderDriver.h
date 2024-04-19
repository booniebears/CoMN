/**
 * @file DecoderDriver.h
 * @author booniebears
 * @brief
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef DECODERDRIVER_H_
#define DECODERDRIVER_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

namespace CoMN {
class DecoderDriver : public BasicUnit {
public:
  DecoderDriver(Param *_param, Technology *_technology);
  virtual ~DecoderDriver() {}
  void Initialize(RowColMode _mode, int _numOfRows, int _numOfColumns,
                  double resMemCellOn);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double _capLoad, double _resLoad,
                        double numRead, double numWrite);
  void CalculatePower(int numReadCells, int numWriteCells, double numRead,
                      double numWrite);

  RowColMode mode;
  int numOfRows;
  int numOfColumns;

private:
  Param *param;
  Technology *tech;
  double widthInvN, widthInvP, widthTgN, widthTgP;
  double resTg;
  double capInvInput, capInvOutput, capTgDrain, capTgGateN, capTgGateP;
};
} // namespace CoMN

#endif