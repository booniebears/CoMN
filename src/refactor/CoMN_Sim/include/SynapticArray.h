/**
 * @file SynapticArray.h
 * @author booniebears
 * @brief
 * @date 2023-10-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef SYNAPTICARRAY_H_
#define SYNAPTICARRAY_H_

#include <vector>

#include "../include/Adder.h"
#include "../include/BasicUnit.h"
#include "../include/CurrentSenseAmp.h"
#include "../include/DFF.h"
#include "../include/DecoderDriver.h"
#include "../include/MultilevelSAEncoder.h"
#include "../include/MultilevelSenseAmp.h"
#include "../include/Mux.h"
#include "../include/MuxDecoder.h"
#include "../include/NewDecoderDriver.h"
#include "../include/NewSwitchMatrix.h"
#include "../include/Precharger.h"
#include "../include/SRAMWriteDriver.h"
#include "../include/SarADC.h"
#include "../include/SenseAmp.h"
#include "../include/ShiftAdder.h"
#include "../include/SwitchMatrix.h"
#include "../include/general/MemCell.h"
#include "../include/general/Param.h"
#include "../include/general/Technology.h"

using namespace std;

namespace CoMN {
class SynapticArray : public BasicUnit {
public:
  SynapticArray(Param *_param, Technology *_technology);
  virtual ~SynapticArray() {}
  void Initialize(int _numOfColumns, int _numOfRows, double _unitWireLength);
  void CalculateArea();
  void CalculateLatency(double _rampInput, vector<double> &colResistance);
  void CalculatePower(vector<double> &colResistance);

  // Parameters of Synaptic Array
  int numOfColumns, numOfRows;
  double widthArray, heightArray, areaArray, readDynamicEnergyArray,
      writeDynamicEnergyArray, writeLatencyArray;
  double unitWireRes;
  int numOfColumnsMux; // how many columns in synaptic array share a mux
  int numOfCellsPerSynapse;
  int numOfReadPulses; // num of read pulses for the input vector = numBitInput
  SpikingMode spikingMode;
  double activityRowRead;  // ratio of rows read, range:[0,1]
  double activityRowWrite; // similar to activityColWrite
  double activityColWrite; // ratio of columns write, range:[0,1]. Not necessary
                           // in training, so it is 0 by default.
  bool isSarADC;           // use SarADC or MLSA?
  bool isCSA;              // MLSA use CSA or VSA?
  int avgWeightBit;        // Average weight for each synapse = cellBit
  double resistanceAvg;
  int numWritePulseAVG;
  int totalNumWritePulse; // Not assigned value at present.

  // Parameters of Ciruit modules
  double areaAG, readLatencyAG, readDynamicEnergyAG;
  double areaADC, areaAccum, areaOther;
  double readLatencyADC, readLatencyAccum, readLatencyOther;
  double readDynamicEnergyADC, readDynamicEnergyAccum, readDynamicEnergyOther;

  // Conditions of NN
  bool conventionalSequential;
  bool conventionalParallel;
  bool BNNsequentialMode;
  bool BNNparallelMode;
  bool XNORsequentialMode;
  bool XNORparallelMode;

  // Cell Properties
  double resCellAccess, capCellAccess; // SRAM cell Resistance/Capacitance
  double capSRAMCell;                  // Capacitance of total SRAMCell(Add up)
  double widthAccessCMOS;
  double resMemCellOn, resMemCellOff, resMemCellAvg;

private:
  Param *param;
  Technology *tech;

  double capWL;  // Capacitance of row (WL for 1T1R and SRAM), Unit: F
  double capBL;  // Capacitance of BL (BL for 1T1R)
  double capCol; // Capacitance of Column
  // double capRowSRAM; // Capacitance of Column
  double resRow, resColumn;

  // Peripheral Ciruit modules
  MuxDecoder wlDecoder; // on the left side
  DecoderDriver wlDecoderDriver;
  NewDecoderDriver wlNewDecoderDriver;
  SwitchMatrix wlSwitchMatrix;
  NewSwitchMatrix wlNewSwitchMatrix;
  SwitchMatrix slSwitchMatrix;
  Mux mux;
  MuxDecoder muxDecoder;
  Precharger precharger;
  SenseAmp senseAmp;
  SRAMWriteDriver sramWriteDriver;
  CurrentSenseAmp rowCurrentSenseAmp;
  DFF dff;     // registers, bind with "adder";
  Adder adder; // adder on the down side
  MultilevelSenseAmp multilevelSenseAmp;
  MultilevelSAEncoder multilevelSAEncoder;
  SarADC sarADC;
  ShiftAdder shiftAdder;

  // Peripheral Circuit modules for Transpose (BP)
  MuxDecoder wlDecoderBP;
  SwitchMatrix wlSwitchMatrixBP;
  Precharger prechargerBP;
  SenseAmp senseAmpBP;
  SRAMWriteDriver sramWriteDriverBP;
  Mux muxBP;
  MuxDecoder muxDecoderBP;
  CurrentSenseAmp rowCurrentSenseAmpBP;
  DFF dffBP;
  Adder adderBP;
  MultilevelSenseAmp multilevelSenseAmpBP;
  MultilevelSAEncoder multilevelSAEncoderBP;
  SarADC sarADCBP;
  ShiftAdder shiftAdderBP;
};
} // namespace CoMN

#endif