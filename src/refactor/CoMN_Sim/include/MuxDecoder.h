/**
 * @file MuxDecoder.h
 * @author booniebears
 * @brief
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef MUXDECODER_H_
#define MUXDECODER_H_

#include "../include/BasicUnit.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"
#include "../include/general/Types.h"

namespace CoMN {
class MuxDecoder : public BasicUnit {
public:
  MuxDecoder(Param *_param, Technology *_technology);
  virtual ~MuxDecoder() {}
  void Initialize(DecoderMode _mode, int _inputBits, bool _isMux,
                  bool _isParallel);
  void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
  void CalculateLatency(double _rampInput, double _capLoad1, double _capLoad2,
                        double numRead, double numWrite);
  void CalculatePower(double numRead, double numWrite);
  double getRampOutput() { return rampOutput; }

  DecoderMode mode;
  int inputBits;   // input bits for a decoder
  bool isMux;      // Decoder for mux, not for WL;
  bool isParallel; // if Parallel, increase MUX Decoder by 8 times

private:
  Param *param;
  Technology *tech;
  // Transistor attributes
  double widthInvN, widthInvP, widthNandN, widthNandP, widthNorN, widthNorP,
      widthDriverInvN, widthDriverInvP;
  int numInv, numNand, numNor, numMetalConnection;
  double capInvInput, capInvOutput, capNandInput, capNandOutput, capNorInput,
      capNorOutput, capDriverInvInput, capDriverInvOutput;
  double rampOutput;
};
} // namespace CoMN

#endif