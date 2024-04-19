/**
 * @file SynapticArray.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <cmath>
#include <iostream>

#include "../include/SynapticArray.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"
using namespace std;

namespace CoMN {
SynapticArray::SynapticArray(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit(),
      wlDecoder(_param, _technology), wlDecoderDriver(_param, _technology),
      wlNewDecoderDriver(_param, _technology),
      wlSwitchMatrix(_param, _technology),
      wlNewSwitchMatrix(_param, _technology),
      slSwitchMatrix(_param, _technology), mux(_param, _technology),
      muxDecoder(_param, _technology), precharger(_param, _technology),
      senseAmp(_param, _technology), sramWriteDriver(_param, _technology),
      rowCurrentSenseAmp(_param, _technology), dff(_param, _technology),
      adder(_param, _technology), multilevelSenseAmp(_param, _technology),
      multilevelSAEncoder(_param, _technology), sarADC(_param, _technology),
      shiftAdder(_param, _technology),
      /* for BP (Transpose SubArray) */
      wlDecoderBP(_param, _technology), wlSwitchMatrixBP(_param, _technology),
      prechargerBP(_param, _technology), senseAmpBP(_param, _technology),
      sramWriteDriverBP(_param, _technology), muxBP(_param, _technology),
      muxDecoderBP(_param, _technology),
      rowCurrentSenseAmpBP(_param, _technology), dffBP(_param, _technology),
      adderBP(_param, _technology), multilevelSenseAmpBP(_param, _technology),
      multilevelSAEncoderBP(_param, _technology), sarADCBP(_param, _technology),
      shiftAdderBP(_param, _technology) {}

/**
 * @brief
 */
void SynapticArray::Initialize(int _numOfColumns, int _numOfRows,
                               double _unitWireResistance) {
  // cout << "(SynapticArray.cpp) In SynapticArray::Initialize!!!" << endl;
  numOfColumns = _numOfColumns;
  numOfRows = _numOfRows;
  unitWireRes = _unitWireResistance;
  // TODO: change the value of param->widthInFeatureSize1T1R?
  // And change technode and featureSize.
  double cellWidth = (param->accessType == CMOS_access)
                         ? param->widthInFeatureSize1T1R
                         : param->widthInFeatureSizeCrossbar;
  double cellHeight = (param->accessType == CMOS_access)
                          ? param->heightInFeatureSize1T1R
                          : param->heightInFeatureSizeCrossbar;
  /* (1) Calculate widthArray and heightArray */
  if (param->memcellType == Type::SRAM) {
    // Not Relax Cell width
    widthArray = numOfColumns * tech->featureSize * cellWidth;
    heightArray = numOfRows * tech->featureSize * cellHeight;
  } else { // RRAM,FeFET
    if (param->accessType == CMOS_access) {
      widthArray = numOfColumns * tech->featureSize * cellWidth;
      heightArray = numOfRows * tech->featureSize * cellHeight;
      // cout << "widthArray = " << widthArray << endl;
      // cout << "heightArray = " << heightArray << endl;
      // cout << "tech->featureSize = " << tech->featureSize << endl;
    } else {
      // Cross-Point
      widthArray = numOfColumns * tech->featureSize * cellWidth;
      heightArray = numOfRows * tech->featureSize * cellHeight;
    }
  }
  /* (2) Calculate relevant Parameters */
  capWL = widthArray * 0.2e-15 / 1e-6; // 0.2 fF/mm
  capBL = widthArray * 0.2e-15 / 1e-6; // capWL = capBL
  capCol = heightArray * 0.2e-15 / 1e-6;
  resRow = widthArray * unitWireRes;
  resColumn = heightArray * unitWireRes;

  /* (3) Initialize relevant modules*/
  if (param->memcellType == Type::SRAM) {
    resCellAccess = CalculateOnResistance(
        param->widthAccessCMOS * tech->featureSize, NMOS, param->temp, tech);
    capCellAccess =
        CalculateDrainCap(param->widthAccessCMOS * tech->featureSize, NMOS,
                          cellWidth * tech->featureSize, tech);
    capSRAMCell =
        capCellAccess +
        CalculateDrainCap(param->widthSRAMCellNMOS * tech->featureSize, NMOS,
                          cellWidth * tech->featureSize, tech) +
        CalculateDrainCap(param->widthSRAMCellPMOS * tech->featureSize, PMOS,
                          cellWidth * tech->featureSize, tech) +
        CalculateGateCap(param->widthSRAMCellNMOS * tech->featureSize, tech) +
        CalculateGateCap(param->widthSRAMCellPMOS * tech->featureSize, tech);

    precharger.Initialize(numOfColumns, resColumn, activityColWrite);
    sramWriteDriver.Initialize(numOfColumns, activityColWrite);
    prechargerBP.Initialize(numOfRows, resRow, activityColWrite);
    sramWriteDriverBP.Initialize(numOfRows, activityColWrite);
    if (conventionalSequential) {
      int numOfAdders = numOfColumns / numOfCellsPerSynapse;
      int numOfBits = ceil(log2(numOfRows)) + 1; // TODO?
      adder.Initialize(numOfBits, numOfAdders);
      dff.Initialize((numOfBits + 1) * numOfAdders);

      if (numOfReadPulses > 1) {
        // Need ShiftAdder to handle multi-bit input
        shiftAdder.Initialize(numOfBits + 1, numOfAdders, numOfReadPulses,
                              spikingMode);
      }
      wlDecoder.Initialize(REGULAR_ROW, (int)(ceil(numOfRows)), false, false);
      senseAmp.Initialize(numOfColumns, false, param->minSenseVoltage,
                          widthArray / numOfColumns);
      // TODO: training condition
    } else if (conventionalParallel) {
      wlSwitchMatrix.Initialize(ROW_MODE, numOfRows, resRow, activityRowRead,
                                activityColWrite, 1);
      int numUnits = ceil(numOfColumns / numOfColumnsMux);
      if (numOfColumnsMux > 1) {
        mux.Initialize(numUnits, numOfColumnsMux,
                       resCellAccess / numOfRows / 2);
        muxDecoder.Initialize(REGULAR_ROW, ceil(log2(numOfColumnsMux)), true,
                              false);
      }

      if (isSarADC) {
        sarADC.Initialize(param->levelOutput, numUnits);
      } else { // MLSA
        multilevelSenseAmp.Initialize(param->levelOutput, numUnits, isCSA,
                                      true);
        multilevelSAEncoder.Initialize(param->levelOutput, numUnits);
      }

      if (numOfReadPulses > 1) {
        shiftAdder.Initialize(log2(param->levelOutput) + 1, numUnits,
                              numOfReadPulses, spikingMode);
      }
      // TODO: training condition
    } else if (BNNsequentialMode || XNORsequentialMode) {
      int numOfAdders = numOfColumns / numOfCellsPerSynapse;
      int numOfBits = ceil(log2(numOfRows)) + avgWeightBit; // TODO?
      adder.Initialize(numOfBits, numOfAdders);
      dff.Initialize((numOfBits + 1) * numOfAdders);
      // No mux and shiftadder needed in BNN and XNOR
      wlDecoder.Initialize(REGULAR_ROW, (int)(ceil(numOfRows)), false, false);
      senseAmp.Initialize(numOfColumns, false, param->minSenseVoltage,
                          widthArray / numOfColumns);
    } else if (BNNparallelMode || XNORparallelMode) {
      wlSwitchMatrix.Initialize(ROW_MODE, numOfRows, resRow, activityRowRead,
                                activityColWrite, 1);
      // No mux and shiftadder needed in BNN and XNOR
      int numUnits = ceil(numOfColumns / numOfColumnsMux);
      if (isSarADC) {
        sarADC.Initialize(param->levelOutput, numUnits);
      } else { // MLSA
        multilevelSenseAmp.Initialize(param->levelOutput, numUnits, isCSA,
                                      true);
        multilevelSAEncoder.Initialize(param->levelOutput, numUnits);
      }
    } else {
      throw runtime_error("operation Mode Error!");
    }
  } else { // RRAM,FeFET
    if (param->accessType == CMOS_access) {
      resCellAccess = param->resistanceOn * IR_DROP_TOLERANCE;
      widthAccessCMOS =
          CalculateOnResistance(tech->featureSize, NMOS, param->temp, tech) *
          LINEAR_REGION_RATIO / resCellAccess; // get access CMOS width
      if (widthAccessCMOS > cellWidth) {
        throw runtime_error(
            "Transistor width of 1T1R is larger than the assigned cell width!");
      }
      resMemCellOn = resCellAccess + param->resistanceOn;
      resMemCellOff = resCellAccess + param->resistanceOff;
      resMemCellAvg = resCellAccess + resistanceAvg;

      capWL += CalculateGateCap(widthAccessCMOS * tech->featureSize, tech) *
               numOfColumns;
      capCol += CalculateDrainCap(widthAccessCMOS * tech->featureSize, NMOS,
                                  cellWidth * tech->featureSize, tech) *
                numOfRows;
    } else {
      // cross-point,consider nonlinearity; TODO: Some properties not calculated
      if (param->nonlinearIV) {
        resMemCellOn = param->resistanceOn;
        resMemCellOff = param->resistanceOff;
        resMemCellAvg = resistanceAvg;
      } else {
        resMemCellOn = param->resistanceOn;
        resMemCellOff = param->resistanceOff;
        resMemCellAvg = resistanceAvg;
      }
    }

    if (conventionalSequential) {
      int numUnits = ceil(numOfColumns / numOfColumnsMux);
      int numOfBits = ceil(log2(numOfRows)) + avgWeightBit; // TODO: ??
      adder.Initialize(numOfBits, numUnits);
      dff.Initialize((numOfBits + 1) * numUnits);
      if (numOfReadPulses > 1) {
        // Need ShiftAdder to handle multi-bit input
        shiftAdder.Initialize(numOfBits + 1, numUnits, numOfReadPulses,
                              spikingMode);
      }
      wlDecoder.Initialize(REGULAR_ROW, (int)(ceil(numOfRows)), false, false);
      // eNVM and FeFET need Decoder Driver
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.Initialize(numOfRows);
      } else {
        wlDecoderDriver.Initialize(ROW_MODE, numOfRows, numOfColumns,
                                   resMemCellOn);
      }
      slSwitchMatrix.Initialize(COL_MODE, numOfColumns, resMemCellOn,
                                activityRowRead, activityColWrite,
                                numWritePulseAVG);
      if (numOfColumnsMux > 1) {
        mux.Initialize(numUnits, numOfColumnsMux, resMemCellOn);
        muxDecoder.Initialize(REGULAR_ROW, ceil(log2(numOfColumnsMux)), true,
                              false);
      }
      int levels = pow(2, avgWeightBit); // = param->levelOutput
      cout << "For RRAM, conventionalSequential: " << endl;
      cout << "levels = " << levels << endl;
      if (isSarADC) {
        sarADC.Initialize(levels, numUnits);
      } else { // MLSA
        multilevelSenseAmp.Initialize(levels, numUnits, isCSA, false);
        if (avgWeightBit > 1) {
          multilevelSAEncoder.Initialize(levels, numUnits);
        }
      }
      // TODO: training condition
    } else if (conventionalParallel) {
      double resTg = resMemCellOn / numOfRows; // TODO: ??? and its usage
      // cout << "resCellAccess = " << resCellAccess << endl;
      // cout << "resistanceOn = " << param->resistanceOn << endl;
      // cout << "resMemCellOn = " << resMemCellOn << endl;
      // cout << "resTg = " << resTg << endl;
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.Initialize(numOfRows, activityRowRead);
      } else {
        wlSwitchMatrix.Initialize(
            ROW_MODE, numOfRows, resTg * numOfRows / numOfColumns,
            activityRowRead, activityColWrite, numWritePulseAVG);
      }
      slSwitchMatrix.Initialize(COL_MODE, numOfColumns, resTg * numOfRows,
                                activityRowRead, activityColWrite,
                                numWritePulseAVG);
      int numUnits = ceil(numOfColumns / numOfColumnsMux);
      if (numOfColumnsMux > 1) {
        // TODO: numOfColumnsMux change to 16?
        mux.Initialize(numUnits, numOfColumnsMux, resTg);
        muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numOfColumnsMux)),
                              true, false);
        // cout << "numOfColumnsMux = " << numOfColumnsMux << endl;
        // cout << "(int)ceil(log2(numOfColumnsMux)) = "
        //      << (int)ceil(log2(numOfColumnsMux)) << endl;
      }
      if (isSarADC) {
        sarADC.Initialize(param->levelOutput, numUnits);
      } else { // MLSA
        multilevelSenseAmp.Initialize(param->levelOutput, numUnits, isCSA,
                                      true);
        multilevelSAEncoder.Initialize(param->levelOutput, numUnits);
      }
      if (numOfReadPulses > 1) {
        // TODO: numOfReadPulses change to 1?
        shiftAdder.Initialize(log2(param->levelOutput) + 1, numUnits,
                              numOfReadPulses, spikingMode);
      }
      // TODO: training condition
    } else if (BNNsequentialMode || XNORsequentialMode) {
      int numUnits = ceil(numOfColumns / numOfColumnsMux);
      int numOfBits = ceil(log2(numOfRows)) + 1; // TODO: ??
      adder.Initialize(numOfBits, numUnits);
      dff.Initialize((numOfBits + 1) * numUnits);
      wlDecoder.Initialize(REGULAR_ROW, (int)(ceil(numOfRows)), false, false);
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.Initialize(numOfRows);
      } else {
        wlDecoderDriver.Initialize(ROW_MODE, numOfRows, numOfColumns,
                                   resMemCellOn);
      }
      slSwitchMatrix.Initialize(COL_MODE, numOfColumns, resMemCellOn,
                                activityRowRead, activityColWrite,
                                numWritePulseAVG);
      if (numOfColumnsMux > 1) {
        mux.Initialize(numUnits, numOfColumnsMux, resMemCellOn);
        muxDecoder.Initialize(REGULAR_ROW, ceil(log2(numOfColumnsMux)), true,
                              false);
      }
      rowCurrentSenseAmp.Initialize(numOfColumns);
    } else if (BNNparallelMode || XNORparallelMode) {
      double resTg = resMemCellOn / numOfRows;
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.Initialize(numOfRows, activityRowRead);
      } else {
        wlSwitchMatrix.Initialize(
            ROW_MODE, numOfRows, resTg * numOfRows / numOfColumns,
            activityRowRead, activityColWrite, numWritePulseAVG);
      }
      slSwitchMatrix.Initialize(COL_MODE, numOfColumns, resTg * numOfRows,
                                activityRowRead, activityColWrite,
                                numWritePulseAVG);
      int numUnits = ceil(numOfColumns / numOfColumnsMux);
      if (numOfColumnsMux > 1) {
        mux.Initialize(numUnits, numOfColumnsMux, resTg);
        muxDecoder.Initialize(REGULAR_ROW, ceil(log2(numOfColumnsMux / 2)),
                              true, true);
      }
      if (isSarADC) {
        sarADC.Initialize(param->levelOutput, numUnits);
      } else { // MLSA
        multilevelSenseAmp.Initialize(param->levelOutput, numUnits, isCSA,
                                      true);
        multilevelSAEncoder.Initialize(param->levelOutput, numUnits);
      }

    } else {
      throw runtime_error("operation Mode Error!");
    }
  }
}

void SynapticArray::CalculateArea() {
  areaArray = widthArray * heightArray;
  if (param->memcellType == Type::SRAM) {
    // Calculate Common device area;
    precharger.CalculateArea(0, widthArray, NONE);
    sramWriteDriver.CalculateArea(0, widthArray, NONE);
    prechargerBP.CalculateArea(heightArray, 0, NONE);
    sramWriteDriverBP.CalculateArea(heightArray, 0, NONE);
    // usedArea, height not final result
    usedArea = precharger.area + sramWriteDriver.area + areaArray;
    width = widthArray;
    height = precharger.height + sramWriteDriver.height + heightArray;

    if (conventionalSequential) {
      adder.CalculateArea(0, widthArray, NONE);
      dff.CalculateArea(0, widthArray, NONE);
      wlDecoder.CalculateArea(heightArray, 0, NONE);
      senseAmp.CalculateArea(0, widthArray, NONE);
      if (numOfReadPulses > 1) {
        shiftAdder.CalculateArea(0, widthArray, NONE);
      }
      usedArea += adder.area + dff.area + wlDecoder.area + senseAmp.area +
                  shiftAdder.area;
      height += adder.height + dff.height + senseAmp.height + shiftAdder.height;
      width += wlDecoder.width;
      // TODO: training condition
      area = height * width;
    } else if (conventionalParallel) {
      wlSwitchMatrix.CalculateArea(heightArray, 0, NONE);
      if (numOfColumnsMux > 1) {
        mux.CalculateArea(0, widthArray, NONE);
        muxDecoder.CalculateArea(0, 0, NONE);
        double minMuxHeight = max(mux.height, muxDecoder.height);
        mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
      }
      if (isSarADC) {
        sarADC.CalculateArea(0, widthArray, NONE);
      } else {
        multilevelSenseAmp.CalculateArea(0, widthArray, NONE);
        multilevelSAEncoder.CalculateArea(0, widthArray, NONE);
      }

      if (numOfReadPulses > 1) {
        shiftAdder.CalculateArea(0, widthArray, NONE);
      }
      cout << "wlSwitchMatrix.area = " << wlSwitchMatrix.area << endl;
      usedArea += wlSwitchMatrix.area + mux.area + muxDecoder.area +
                  sarADC.area + multilevelSenseAmp.area +
                  multilevelSAEncoder.area + shiftAdder.area;
      // TODO: muxDecoder && mux in SRAM array???
      width += max(wlSwitchMatrix.width, muxDecoder.width);
      height += sarADC.height + multilevelSenseAmp.height +
                multilevelSAEncoder.height + shiftAdder.height + mux.height;
      // TODO: training condition
      area = height * width;
    } else if (BNNsequentialMode || XNORsequentialMode) {
      adder.CalculateArea(0, widthArray, NONE);
      dff.CalculateArea(0, widthArray, NONE);
      wlDecoder.CalculateArea(heightArray, 0, NONE);
      senseAmp.CalculateArea(0, widthArray, NONE);
      usedArea += adder.area + dff.area + wlDecoder.area + senseAmp.area;
      width += wlDecoder.width;
      height += adder.area + dff.area + senseAmp.area;
      area = height * width;
    } else if (BNNparallelMode || XNORparallelMode) {
      wlSwitchMatrix.CalculateArea(heightArray, 0, NONE);
      if (isSarADC) {
        sarADC.CalculateArea(0, widthArray, NONE);
      } else {
        multilevelSenseAmp.CalculateArea(0, widthArray, NONE);
        multilevelSAEncoder.CalculateArea(0, widthArray, NONE);
      }
      usedArea +=
          sarADC.area + multilevelSenseAmp.area + multilevelSAEncoder.area;
      width += wlSwitchMatrix.width;
      height += sarADC.height + multilevelSenseAmp.height +
                multilevelSAEncoder.height;
      area = height * width;
    } else {
      throw runtime_error("operation Mode Error!");
    }
  } else {
    // RRAM,FeFET
    if (conventionalSequential) {
      adder.CalculateArea(0, widthArray, NONE);
      dff.CalculateArea(0, widthArray, NONE);
      if (numOfReadPulses > 1) {
        shiftAdder.CalculateArea(0, widthArray, NONE);
      }
      wlDecoder.CalculateArea(0, heightArray, NONE);
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.CalculateArea(heightArray, 0, NONE);
      } else {
        wlDecoderDriver.CalculateArea(heightArray, 0, NONE);
      }
      slSwitchMatrix.CalculateArea(0, widthArray, NONE);
      if (numOfColumnsMux > 1) {
        mux.CalculateArea(0, widthArray, NONE);
        muxDecoder.CalculateArea(0, 0, NONE);
        double minMuxHeight = max(mux.height, muxDecoder.height);
        mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
      }
      if (isSarADC) {
        sarADC.CalculateArea(0, widthArray, NONE);
      } else {
        multilevelSenseAmp.CalculateArea(0, widthArray, NONE);
        if (avgWeightBit > 1) {
          multilevelSAEncoder.CalculateArea(0, widthArray, NONE);
        }
      }
      usedArea = areaArray + adder.area + dff.area + shiftAdder.area +
                 wlDecoder.area + wlNewDecoderDriver.area +
                 wlDecoderDriver.area + slSwitchMatrix.area + mux.area +
                 muxDecoder.area + sarADC.area + multilevelSenseAmp.area +
                 multilevelSAEncoder.area;
      width = widthArray + max(wlNewDecoderDriver.width +
                                   wlDecoderDriver.width + wlDecoder.width,
                               muxDecoder.width);
      height = heightArray + slSwitchMatrix.height + mux.height +
               multilevelSenseAmp.height + multilevelSAEncoder.height +
               adder.height + dff.height + shiftAdder.height + sarADC.height;
      // TODO: training condition
      area = width * height;
    } else if (conventionalParallel) {
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.CalculateArea(heightArray, 0, NONE);
      } else {
        wlSwitchMatrix.CalculateArea(heightArray, 0, NONE);
      }
      slSwitchMatrix.CalculateArea(0, widthArray, NONE);
      if (numOfColumnsMux > 1) {
        mux.CalculateArea(0, widthArray, NONE);
        muxDecoder.CalculateArea(0, 0, NONE);
        // TODO: Why Calculated in this way ?
        // double minMuxHeight = max(mux.height, muxDecoder.height);
        // mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
      }
      if (isSarADC) {
        sarADC.CalculateArea(0, widthArray, NONE);
      } else {
        multilevelSenseAmp.CalculateArea(0, widthArray, NONE);
        multilevelSAEncoder.CalculateArea(0, widthArray, NONE);
      }
      if (numOfReadPulses > 1) {
        // TODO: shiftAddInput && shiftAddWeight -> shiftAdder?
        shiftAdder.CalculateArea(0, widthArray, NONE);
      }
      // cout << "************* usedArea Calculate Begins *************" << endl;
      // cout << "areaArray = " << areaArray << endl;
      // cout << "wlNewSwitchMatrix.area = " << wlNewSwitchMatrix.area << endl;
      // cout << "wlSwitchMatrix.area = " << wlSwitchMatrix.area << endl;
      // cout << "slSwitchMatrix.area = " << slSwitchMatrix.area << endl;
      // cout << "mux.area = " << mux.area << endl;
      // cout << "muxDecoder.area = " << muxDecoder.area << endl;
      // cout << "sarADC.area = " << sarADC.area << endl;
      // cout << "multilevelSenseAmp.area = " << multilevelSenseAmp.area << endl;
      // cout << "multilevelSAEncoder.area = " << multilevelSAEncoder.area << endl;
      // cout << "shiftAdder.area = " << shiftAdder.area << endl;
      usedArea = areaArray + wlNewSwitchMatrix.area + wlSwitchMatrix.area +
                 slSwitchMatrix.area + mux.area + muxDecoder.area +
                 sarADC.area + multilevelSenseAmp.area +
                 multilevelSAEncoder.area + shiftAdder.area;
      // cout << "usedArea = " << usedArea << endl;
      // cout << "************* usedArea Calculate Ends *************" << endl;
      width = widthArray + max(wlSwitchMatrix.width + wlNewSwitchMatrix.width,
                               muxDecoder.width);
      height = heightArray + slSwitchMatrix.height + mux.height +
               sarADC.height + multilevelSenseAmp.height +
               multilevelSAEncoder.height + shiftAdder.height;
      // TODO: training condition
      area = width * height;
    } else if (BNNsequentialMode || XNORsequentialMode) {
      adder.CalculateArea(0, widthArray, NONE);
      dff.CalculateArea(0, widthArray, NONE);
      wlDecoder.CalculateArea(0, heightArray, NONE);
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.CalculateArea(heightArray, 0, NONE);
      } else {
        wlDecoderDriver.CalculateArea(heightArray, 0, NONE);
      }
      slSwitchMatrix.CalculateArea(0, widthArray, NONE);
      if (numOfColumnsMux > 1) {
        mux.CalculateArea(0, widthArray, NONE);
        muxDecoder.CalculateArea(0, 0, NONE);
        double minMuxHeight = max(mux.height, muxDecoder.height);
        mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
      }
      rowCurrentSenseAmp.CalculateArea(0, widthArray, NONE);
      usedArea = areaArray + adder.area + dff.area + wlDecoder.area +
                 wlNewDecoderDriver.area + wlDecoderDriver.area +
                 slSwitchMatrix.area + mux.area + muxDecoder.area +
                 rowCurrentSenseAmp.area;
      width = widthArray + max(wlDecoder.width + wlNewDecoderDriver.width +
                                   wlDecoderDriver.width,
                               muxDecoder.width);
      height = heightArray + slSwitchMatrix.height + mux.height + adder.height +
               dff.height + rowCurrentSenseAmp.height;
      area = width * height;
    } else if (BNNparallelMode || XNORparallelMode) {
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.CalculateArea(heightArray, 0, NONE);
      } else {
        wlSwitchMatrix.CalculateArea(heightArray, 0, NONE);
      }
      slSwitchMatrix.CalculateArea(0, widthArray, NONE);
      if (numOfColumnsMux > 1) {
        mux.CalculateArea(0, widthArray, NONE);
        muxDecoder.CalculateArea(0, 0, NONE);
        double minMuxHeight = max(mux.height, muxDecoder.height);
        mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
      }
      if (isSarADC) {
        sarADC.CalculateArea(0, widthArray, NONE);
      } else {
        multilevelSenseAmp.CalculateArea(0, widthArray, NONE);
        multilevelSAEncoder.CalculateArea(0, widthArray, NONE);
      }

      usedArea = areaArray + wlNewSwitchMatrix.area + wlSwitchMatrix.area +
                 slSwitchMatrix.area + mux.area + muxDecoder.area +
                 sarADC.area + multilevelSenseAmp.area +
                 multilevelSAEncoder.area;
      width = widthArray + max(wlSwitchMatrix.width + wlNewSwitchMatrix.width,
                               muxDecoder.width);
      height = heightArray + slSwitchMatrix.height + mux.height +
               sarADC.height + multilevelSenseAmp.height +
               multilevelSAEncoder.height;
      area = width * height;
    }
  }
}

void SynapticArray::CalculateLatency(double _rampInput,
                                     vector<double> &colResistance) {
  if (param->memcellType == Type::SRAM) {
    // TODO: simplify numReadOperationPerRow to 1; activityRowRead = 1,
    // numWrite = activityColWrite = 0
    if (conventionalSequential) {
      double numRead = numOfRows * activityRowRead;
      double numWrite = numOfRows * activityRowWrite;
      dff.CalculateLatency(numRead);
      adder.CalculateLatency(INF_RAMP, dff.getCap(), numRead);
      if (numOfReadPulses > 1) {
        // Need ShiftAdder to handle multi-bit input
        shiftAdder.CalculateLatency(1);
      }
      wlDecoder.CalculateLatency(INF_RAMP, capWL, 0, numRead, numWrite);
      senseAmp.CalculateLatency(numRead);
      precharger.CalculateLatency(capCol, numRead, numWrite);
      sramWriteDriver.CalculateLatency(INF_RAMP, capCol, resColumn, numWrite);
      // TODO: training condition

      // Read
      readLatency += dff.readLatency + adder.readLatency +
                     shiftAdder.readLatency + wlDecoder.readLatency +
                     senseAmp.readLatency + precharger.readLatency;

      double widthNMOS = param->widthSRAMCellNMOS * tech->featureSize;
      double widthPMOS = param->widthSRAMCellPMOS * tech->featureSize;
      double resPullDown =
          CalculateOnResistance(widthNMOS, NMOS, param->temp, tech);
      double tauf = (resCellAccess + resPullDown) * (capCellAccess + capCol) +
                    resColumn * capCol / 2;
      tauf *= log(tech->vdd / (tech->vdd - param->minSenseVoltage / 2));
      double gm = CalculateTransconductance(widthAccessCMOS * tech->featureSize,
                                            NMOS, tech);
      double beta = 1 / (gm * resPullDown);
      double colReadLatency =
          horowitz(tauf, beta, wlDecoder.getRampOutput(), nullptr) * numRead;
      readLatency += colReadLatency;

      // Write (the average delay of pullup and pulldown inverter in SRAM cell)
      writeLatency += wlDecoder.writeLatency + precharger.writeLatency +
                      sramWriteDriver.writeLatency;

      double resPull =
          (CalculateOnResistance(widthNMOS, NMOS, param->temp, tech) +
           CalculateOnResistance(widthPMOS, PMOS, param->temp, tech)) /
          2;
      tauf = resPull * capSRAMCell;
      gm = (CalculateTransconductance(widthNMOS, NMOS, tech) +
            CalculateTransconductance(widthPMOS, PMOS, tech)) /
           2;
      beta = 1 / (resPull * gm);
      double colWriteLatency =
          horowitz(tauf, beta, INF_RAMP, nullptr) * numWrite;
      writeLatency += colWriteLatency;
    } else if (conventionalParallel) {
      // numRead Switch to numOfColumnsMux in Parallel mode
      double numRead = numOfColumnsMux;
      double numWrite = numOfRows * activityRowWrite;
      precharger.CalculateLatency(capCol, numRead, numWrite);
      sramWriteDriver.CalculateLatency(INF_RAMP, capCol, resColumn, numWrite);
      wlSwitchMatrix.CalculateLatency(INF_RAMP, capWL, resRow, numRead,
                                      2 * numWrite); // TODO: *2?
      if (numOfColumnsMux > 1) {
        mux.CalculateLatency(0, numRead);
        int numUnits = ceil(numOfColumns / numOfColumnsMux);
        muxDecoder.CalculateLatency(INF_RAMP, mux.capTgGateN * numUnits,
                                    mux.capTgGateP * numUnits, numRead, 0);
      }

      if (isSarADC) {
        sarADC.CalculateLatency(numRead);
      } else { // MLSA
        multilevelSenseAmp.CalculateLatency(colResistance, numRead);
        multilevelSAEncoder.CalculateLatency(INF_RAMP, numRead);
      }

      if (numOfReadPulses > 1) {
        shiftAdder.CalculateLatency(numRead);
      }
      // TODO: training condition

      // Read
      readLatency += precharger.readLatency + wlSwitchMatrix.readLatency +
                     sarADC.readLatency + multilevelSenseAmp.readLatency +
                     multilevelSAEncoder.readLatency + shiftAdder.readLatency;

      double widthNMOS = param->widthSRAMCellNMOS * tech->featureSize;
      double widthPMOS = param->widthSRAMCellPMOS * tech->featureSize;
      double resPullDown =
          CalculateOnResistance(widthNMOS, NMOS, param->temp, tech);
      double tauf = (resCellAccess + resPullDown) * (capCellAccess + capCol) +
                    resColumn * capCol / 2;

      tauf *= log(tech->vdd / (tech->vdd - param->minSenseVoltage / 2));
      double gm = CalculateTransconductance(widthAccessCMOS * tech->featureSize,
                                            NMOS, tech);
      double beta = 1 / (resPullDown * gm);
      double colReadLatency =
          horowitz(tauf, beta, wlSwitchMatrix.getRampOutput(), nullptr);
      readLatency += colReadLatency;
      double muxLatency =
          ((numOfColumnsMux > 1) ? (mux.readLatency + muxDecoder.readLatency)
                                 : 0) /
          numOfReadPulses;
      // TODO: Why use max??
      readLatency += max(wlSwitchMatrix.readLatency, muxLatency);

      // Write
      writeLatency = wlSwitchMatrix.writeLatency + precharger.writeLatency +
                     sramWriteDriver.writeLatency;
      double resPull =
          (CalculateOnResistance(widthNMOS, NMOS, param->temp, tech) +
           CalculateOnResistance(widthPMOS, PMOS, param->temp, tech)) /
          2; // take average
      tauf = resPull * capSRAMCell;
      gm = (CalculateTransconductance(widthNMOS, NMOS, tech) +
            CalculateTransconductance(widthPMOS, PMOS, tech)) /
           2;
      beta = 1 / (resPull * gm);
      double colWriteLatency =
          horowitz(tauf, beta, INF_RAMP, nullptr) * numWrite;
      writeLatency += colWriteLatency;

    } else if (BNNsequentialMode || XNORsequentialMode) {
      double numRead = numOfRows * activityRowRead;
      double numWrite = numOfRows * activityRowWrite;
      dff.CalculateLatency(numRead);
      adder.CalculateLatency(INF_RAMP, dff.getCap(), numRead);
      wlDecoder.CalculateLatency(INF_RAMP, capWL, 0, numRead, numWrite);
      senseAmp.CalculateLatency(numRead);
      precharger.CalculateLatency(capCol, numRead, numWrite);
      sramWriteDriver.CalculateLatency(INF_RAMP, capCol, resColumn, numWrite);

      // Read
      readLatency += dff.readLatency + adder.readLatency +
                     wlDecoder.readLatency + senseAmp.readLatency +
                     precharger.readLatency;

      double widthNMOS = param->widthSRAMCellNMOS * tech->featureSize;
      double widthPMOS = param->widthSRAMCellPMOS * tech->featureSize;
      double resPullDown =
          CalculateOnResistance(widthNMOS, NMOS, param->temp, tech);
      double tauf = (resCellAccess + resPullDown) * (capCellAccess + capCol) +
                    resColumn * capCol / 2;
      tauf *= log(tech->vdd / (tech->vdd - param->minSenseVoltage / 2));
      double gm = CalculateTransconductance(widthAccessCMOS * tech->featureSize,
                                            NMOS, tech);
      double beta = 1 / (gm * resPullDown);
      double colReadLatency =
          horowitz(tauf, beta, wlDecoder.getRampOutput(), nullptr) * numRead;
      readLatency += colReadLatency;

      // Write
      writeLatency += wlDecoder.writeLatency + precharger.writeLatency +
                      sramWriteDriver.writeLatency;
      double resPull =
          (CalculateOnResistance(widthNMOS, NMOS, param->temp, tech) +
           CalculateOnResistance(widthPMOS, PMOS, param->temp, tech)) /
          2;
      tauf = resPull * capSRAMCell;
      gm = (CalculateTransconductance(widthNMOS, NMOS, tech) +
            CalculateTransconductance(widthPMOS, PMOS, tech)) /
           2;
      beta = 1 / (resPull * gm);
      double colWriteLatency =
          horowitz(tauf, beta, INF_RAMP, nullptr) * numWrite;
      writeLatency += colWriteLatency;

    } else if (BNNparallelMode || XNORparallelMode) {
      double numRead = numOfColumnsMux;
      double numWrite = numOfRows * activityRowWrite;
      precharger.CalculateLatency(capCol, numRead, numWrite);
      sramWriteDriver.CalculateLatency(INF_RAMP, capCol, resColumn, numWrite);
      wlSwitchMatrix.CalculateLatency(INF_RAMP, capWL, resRow, numRead,
                                      2 * numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculateLatency(0, numRead);
        int numUnits = ceil(numOfColumns / numOfColumnsMux);
        muxDecoder.CalculateLatency(INF_RAMP, mux.capTgGateN * numUnits,
                                    mux.capTgGateP * numUnits, numRead, 0);
      }

      if (isSarADC) {
        sarADC.CalculateLatency(numRead);
      } else { // MLSA
        multilevelSenseAmp.CalculateLatency(colResistance, numRead);
        multilevelSAEncoder.CalculateLatency(INF_RAMP, numRead);
      }

      // Read
      readLatency += precharger.readLatency + wlSwitchMatrix.readLatency +
                     sarADC.readLatency + multilevelSenseAmp.readLatency +
                     multilevelSAEncoder.readLatency;

      double widthNMOS = param->widthSRAMCellNMOS * tech->featureSize;
      double widthPMOS = param->widthSRAMCellPMOS * tech->featureSize;
      double resPullDown =
          CalculateOnResistance(widthNMOS, NMOS, param->temp, tech);
      double tauf = (resCellAccess + resPullDown) * (capCellAccess + capCol) +
                    resColumn * capCol / 2;

      tauf *= log(tech->vdd / (tech->vdd - param->minSenseVoltage / 2));
      double gm = CalculateTransconductance(widthAccessCMOS * tech->featureSize,
                                            NMOS, tech);
      double beta = 1 / (resPullDown * gm);
      double colReadLatency =
          horowitz(tauf, beta, wlSwitchMatrix.getRampOutput(), nullptr);
      readLatency += colReadLatency;
      double muxLatency =
          ((numOfColumnsMux > 1) ? (mux.readLatency + muxDecoder.readLatency)
                                 : 0) /
          numOfReadPulses;
      readLatency += max(wlSwitchMatrix.readLatency, muxLatency);

      // Write
      writeLatency = wlSwitchMatrix.writeLatency + precharger.writeLatency +
                     sramWriteDriver.writeLatency;
      double resPull =
          (CalculateOnResistance(widthNMOS, NMOS, param->temp, tech) +
           CalculateOnResistance(widthPMOS, PMOS, param->temp, tech)) /
          2; // take average
      tauf = resPull * capSRAMCell;
      gm = (CalculateTransconductance(widthNMOS, NMOS, tech) +
            CalculateTransconductance(widthPMOS, PMOS, tech)) /
           2;
      beta = 1 / (resPull * gm);
      double colWriteLatency =
          horowitz(tauf, beta, INF_RAMP, nullptr) * numWrite;
      writeLatency += colWriteLatency;

    } else {
      throw runtime_error("operation Mode Error!");
    }
  } else {
    // RRAM,FeFET
    if (conventionalSequential) {
      // Start
      double tauf = capCol * resMemCellAvg;
      // assume the 15~20% voltage drop is enough for sensing
      double colRamp, colLatency = tauf * 0.2 * numOfColumnsMux;
      horowitz(tauf, 0, INF_RAMP, &colRamp); // get colRamp

      // TODO: why * 2?? why * numOfColumnsMux??
      double numRead = numOfRows * activityRowRead * numOfColumnsMux;
      double numWrite = numOfRows * activityRowWrite * 2;
      dff.CalculateLatency(numRead);
      adder.CalculateLatency(INF_RAMP, dff.getCap(), numRead);
      if (numOfReadPulses > 1) {
        // shift for numOfColumnsMux times
        shiftAdder.CalculateLatency(numOfColumnsMux);
      }
      wlDecoder.CalculateLatency(INF_RAMP, capWL, 0, numRead, numWrite);
      // eNVM and FeFET need Decoder Driver
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.CalculateLatency(wlDecoder.getRampOutput(), capWL,
                                            resRow, numRead, numWrite);
      } else {
        wlDecoderDriver.CalculateLatency(wlDecoder.getRampOutput(), capBL,
                                         resRow, numRead, numWrite);
      }
      slSwitchMatrix.CalculateLatency(INF_RAMP, capCol, resColumn, 0, numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculateLatency(0, numOfColumnsMux);
        int numUnits = ceil(numOfColumns / numOfColumnsMux);
        muxDecoder.CalculateLatency(INF_RAMP, mux.capTgGateN * numUnits,
                                    mux.capTgGateP * numUnits, numOfColumnsMux,
                                    0);
      }
      if (isSarADC) {
        sarADC.CalculateLatency(numRead);
      } else { // MLSA
        multilevelSenseAmp.CalculateLatency(colResistance, numRead);
        if (avgWeightBit > 1) {
          multilevelSAEncoder.CalculateLatency(INF_RAMP, numRead);
        }
      }

      // TODO: training condition
      readLatency +=
          dff.readLatency + adder.readLatency + shiftAdder.readLatency +
          sarADC.readLatency + multilevelSenseAmp.readLatency +
          multilevelSAEncoder.readLatency + colLatency / numOfReadPulses;

      double muxLatency =
          ((numOfColumnsMux > 1) ? (mux.readLatency + muxDecoder.readLatency)
                                 : 0);
      muxLatency /= numOfReadPulses;
      readLatency +=
          max(muxLatency, wlDecoder.readLatency + wlDecoderDriver.readLatency +
                              wlNewDecoderDriver.readLatency);

      writeLatency += totalNumWritePulse * param->writePulseWidth;
      writeLatency +=
          max(wlDecoder.writeLatency + wlNewDecoderDriver.writeLatency +
                  wlDecoderDriver.writeLatency,
              slSwitchMatrix.writeLatency);
    } else if (conventionalParallel) {
      double tauf = capCol * resMemCellAvg / (numOfRows / 2);
      // assume the 15~20% voltage drop is enough for sensing
      double colRamp, colLatency = tauf * 0.2 * numOfColumnsMux;
      horowitz(tauf, 0, INF_RAMP, &colRamp); // get colRamp

      double numRead = numOfColumnsMux;
      double numWrite = numOfRows * activityRowWrite * 2;
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.CalculateLatency(INF_RAMP, capWL, resRow, numRead,
                                           numWrite);
      } else {
        wlSwitchMatrix.CalculateLatency(INF_RAMP, capBL, resRow, numRead,
                                        numWrite);
      }
      slSwitchMatrix.CalculateLatency(INF_RAMP, capCol, resColumn, 0, numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculateLatency(0, numRead);
        int numUnits = ceil(numOfColumns / numOfColumnsMux);
        muxDecoder.CalculateLatency(INF_RAMP, mux.capTgGateN * numUnits,
                                    mux.capTgGateP * numUnits, numRead, 0);
      }
      if (isSarADC) {
        sarADC.CalculateLatency(numRead);
      } else { // MLSA
        multilevelSenseAmp.CalculateLatency(colResistance, numRead);
        multilevelSAEncoder.CalculateLatency(INF_RAMP, numRead);
      }
      if (numOfReadPulses > 1) {
        shiftAdder.CalculateLatency(numRead);
      }

      // TODO: training condition

      // Read
      readLatency += sarADC.readLatency + multilevelSenseAmp.readLatency +
                     multilevelSAEncoder.readLatency + shiftAdder.readLatency +
                     colLatency / numOfReadPulses;

      // TODO: different from the situation in Sequential??
      double muxLatency = ((numOfColumnsMux > 1) ? mux.readLatency : 0);
      muxLatency /= numOfReadPulses;
      readLatency +=
          max(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency,
              muxLatency);
      // Write
      writeLatency += totalNumWritePulse * param->writePulseWidth;
      writeLatency +=
          max(wlSwitchMatrix.writeLatency + wlNewSwitchMatrix.writeLatency,
              slSwitchMatrix.writeLatency);

      // cout << "*********** Read Latency Calculate Begins ***********" << endl;
      // cout << "readLatency = " << readLatency << endl;
      // cout << "multilevelSenseAmp.readLatency = "
      //      << multilevelSenseAmp.readLatency << endl;
      // cout << "multilevelSAEncoder.readLatency = "
      //      << multilevelSAEncoder.readLatency << endl;
      // cout << "shiftAdd.readLatency = " << shiftAdder.readLatency << endl;
      // cout << "colLatency = " << colLatency << endl;
      // cout << "sarADC.readLatency = " << sarADC.readLatency << endl;
      // cout << "mux.readLatency = " << mux.readLatency << endl;
      // cout << "SwitchMatrix readLatency = " << wlNewSwitchMatrix.readLatency
      //      << endl;
      // cout << "*********** Read Latency Calculate Ends ***********" << endl;

      // cout << "*********** Write Latency Calculate Begins ***********" << endl;
      // cout << "writeLatency = " << writeLatency << endl;
      // cout << "wlNewSwitchMatrix.area = " << wlNewSwitchMatrix.area << endl;
      // cout << "*********** Write Latency Calculate Ends ***********" << endl;

    } else if (BNNsequentialMode || XNORsequentialMode) {
      double tauf = capCol * resMemCellAvg;
      // assume the 15~20% voltage drop is enough for sensing
      double colRamp, colLatency = tauf * 0.2 * numOfColumnsMux;
      horowitz(tauf, 0, INF_RAMP, &colRamp); // get colRamp

      double numRead = numOfRows * activityRowRead * numOfColumnsMux;
      double numWrite = numOfRows * activityRowWrite * 2;
      dff.CalculateLatency(numRead);
      adder.CalculateLatency(INF_RAMP, dff.getCap(), numRead);

      wlDecoder.CalculateLatency(INF_RAMP, capWL, 0, numRead, numWrite);
      // eNVM and FeFET need Decoder Driver
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.CalculateLatency(wlDecoder.getRampOutput(), capWL,
                                            resRow, numRead, numWrite);
      } else {
        wlDecoderDriver.CalculateLatency(wlDecoder.getRampOutput(), capBL,
                                         resRow, numRead, numWrite);
      }
      slSwitchMatrix.CalculateLatency(INF_RAMP, capCol, resColumn, 0, numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculateLatency(0, numOfColumnsMux);
        int numUnits = ceil(numOfColumns / numOfColumnsMux);
        muxDecoder.CalculateLatency(INF_RAMP, mux.capTgGateN * numUnits,
                                    mux.capTgGateP * numUnits, numOfColumnsMux,
                                    0);
      }
      rowCurrentSenseAmp.CalculateLatency(
          colResistance, numOfColumnsMux * numOfRows * activityRowRead);
      readLatency += dff.readLatency + adder.readLatency +
                     rowCurrentSenseAmp.readLatency +
                     colLatency / numOfReadPulses;
      double muxLatency =
          ((numOfColumnsMux > 1) ? (mux.readLatency + muxDecoder.readLatency)
                                 : 0);
      muxLatency /= numOfReadPulses;
      readLatency +=
          max(muxLatency, wlDecoder.readLatency + wlDecoderDriver.readLatency +
                              wlNewDecoderDriver.readLatency);

      writeLatency += totalNumWritePulse * param->writePulseWidth;
      writeLatency +=
          max(wlDecoder.writeLatency + wlNewDecoderDriver.writeLatency +
                  wlDecoderDriver.writeLatency,
              slSwitchMatrix.writeLatency);
    } else if (BNNparallelMode || XNORparallelMode) {
      double tauf = capCol * resMemCellAvg / (numOfRows / 2);
      // assume the 15~20% voltage drop is enough for sensing
      double colRamp, colLatency = tauf * 0.2 * numOfColumnsMux;
      horowitz(tauf, 0, INF_RAMP, &colRamp); // get colRamp

      double numRead = numOfColumnsMux;
      double numWrite = numOfRows * activityRowWrite * 2;
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.CalculateLatency(INF_RAMP, capWL, resRow, numRead,
                                           numWrite);
      } else {
        wlSwitchMatrix.CalculateLatency(INF_RAMP, capBL, resRow, numRead,
                                        numWrite);
      }
      slSwitchMatrix.CalculateLatency(INF_RAMP, capCol, resColumn, 0, numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculateLatency(0, numRead);
        int numUnits = ceil(numOfColumns / numOfColumnsMux);
        muxDecoder.CalculateLatency(INF_RAMP, mux.capTgGateN * numUnits,
                                    mux.capTgGateP * numUnits, numRead, 0);
      }
      if (isSarADC) {
        sarADC.CalculateLatency(numRead);
      } else { // MLSA
        multilevelSenseAmp.CalculateLatency(colResistance, numRead);
        multilevelSAEncoder.CalculateLatency(INF_RAMP, numRead);
      }

      // Read
      readLatency += sarADC.readLatency + multilevelSenseAmp.readLatency +
                     multilevelSAEncoder.readLatency +
                     colLatency / numOfReadPulses;

      double muxLatency = ((numOfColumnsMux > 1) ? mux.readLatency : 0);
      muxLatency /= numOfReadPulses;
      readLatency +=
          max(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency,
              muxLatency);
      // Write
      writeLatency += totalNumWritePulse * param->writePulseWidth;
      writeLatency +=
          max(wlSwitchMatrix.writeLatency + wlNewSwitchMatrix.writeLatency,
              slSwitchMatrix.writeLatency);

    } else {
      throw runtime_error("operation Mode Error!");
    }
  }
}

void SynapticArray::CalculatePower(vector<double> &colResistance) {
  if (param->memcellType == Type::SRAM) {
    // Array leakage (assume 2 INV)
    leakage = CalculateGateLeakage(INV, 1,
                                   param->widthSRAMCellNMOS * tech->featureSize,
                                   param->widthSRAMCellPMOS * tech->featureSize,
                                   param->temp, tech) *
              tech->vdd * 2;
    leakage *= numOfRows * numOfColumns;
    if (conventionalSequential) {
      double numRead = numOfRows * activityRowRead;
      double numWrite = numOfRows * activityRowWrite;
      precharger.CalculatePower(numRead, numWrite);
      sramWriteDriver.CalculatePower(numWrite);
      adder.CalculatePower(numRead, numOfColumns / numOfCellsPerSynapse);
      dff.CalculatePower(numRead, numOfColumns / numOfCellsPerSynapse *
                                      (adder.numOfBits + 1));
      if (numOfReadPulses > 1) {
        // Need ShiftAdder to handle multi-bit input
        shiftAdder.CalculatePower(numRead);
      }
      wlDecoder.CalculatePower(numRead, numWrite);
      senseAmp.CalculatePower(numRead);
      // TODO: training condition

      // TODO: The calculation of array energy???
      double readDynamicEnergyArray = 0; // Just BL charging
      double writeDynamicEnergyArray =
          capSRAMCell * tech->vdd * tech->vdd * 2 * numOfColumns *
          activityColWrite * numOfRows * activityRowWrite; // flip Q and Q_bar

      readDynamicEnergy +=
          precharger.readDynamicEnergy + readDynamicEnergyArray +
          adder.readDynamicEnergy + dff.readDynamicEnergy +
          shiftAdder.readDynamicEnergy + wlDecoder.readDynamicEnergy +
          senseAmp.readDynamicEnergy;
      writeDynamicEnergy +=
          precharger.writeDynamicEnergy + sramWriteDriver.writeDynamicEnergy +
          wlDecoder.writeDynamicEnergy + writeDynamicEnergyArray;
      leakage += precharger.leakage + sramWriteDriver.leakage + adder.leakage +
                 dff.leakage + shiftAdder.leakage + wlDecoder.leakage +
                 senseAmp.leakage;

    } else if (conventionalParallel) {
      // TODO:numRead && numWrite don't match those in CalculateLatency???
      precharger.CalculatePower(numOfColumnsMux, numOfRows * activityRowWrite);
      sramWriteDriver.CalculatePower(numOfRows * activityRowWrite);
      wlSwitchMatrix.CalculatePower(numOfColumnsMux,
                                    2 * numOfRows * activityRowWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculatePower(numOfColumnsMux);
        muxDecoder.CalculatePower(numOfColumnsMux, 1);
      }
      if (isSarADC) {
        sarADC.CalculatePower(colResistance, 1);
      } else { // MLSA
        multilevelSenseAmp.CalculatePower(colResistance, 1);
        // Start From Here:
        multilevelSAEncoder.CalculatePower(numOfColumnsMux);
      }
      if (numOfReadPulses > 1) {
        shiftAdder.CalculatePower(numOfColumnsMux);
      }
      // TODO: training condition
      double readDynamicEnergyArray = 0; // Just BL charging
      double writeDynamicEnergyArray =
          capSRAMCell * tech->vdd * tech->vdd * 2 * numOfColumns *
          activityColWrite * numOfRows * activityRowWrite; // flip Q and Q_bar
      double muxReadEnergy =
          (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy) /
          numOfReadPulses;
      readDynamicEnergy +=
          precharger.readDynamicEnergy + wlSwitchMatrix.readDynamicEnergy +
          muxReadEnergy + sarADC.readDynamicEnergy + readDynamicEnergyArray +
          multilevelSenseAmp.readDynamicEnergy +
          multilevelSAEncoder.readDynamicEnergy + shiftAdder.readDynamicEnergy;
      writeDynamicEnergy +=
          precharger.writeDynamicEnergy + sramWriteDriver.writeDynamicEnergy +
          wlSwitchMatrix.writeDynamicEnergy + writeDynamicEnergyArray;
      leakage += precharger.leakage + sramWriteDriver.leakage +
                 wlSwitchMatrix.leakage + multilevelSAEncoder.leakage +
                 shiftAdder.leakage;

    } else if (BNNsequentialMode || XNORsequentialMode) {
      double numRead = numOfRows * activityRowRead;
      double numWrite = numOfRows * activityRowWrite;
      precharger.CalculatePower(numRead, numWrite);
      sramWriteDriver.CalculatePower(numWrite);
      adder.CalculatePower(numRead, numOfColumns / numOfCellsPerSynapse);
      dff.CalculatePower(numRead, numOfColumns / numOfCellsPerSynapse *
                                      (adder.numOfBits + 1));

      wlDecoder.CalculatePower(numRead, numWrite);
      senseAmp.CalculatePower(numRead);
      double readDynamicEnergyArray = 0; // Just BL charging
      double writeDynamicEnergyArray =
          capSRAMCell * tech->vdd * tech->vdd * 2 * numOfColumns *
          activityColWrite * numOfRows * activityRowWrite; // flip Q and Q_bar
      readDynamicEnergy += precharger.readDynamicEnergy +
                           readDynamicEnergyArray + adder.readDynamicEnergy +
                           dff.readDynamicEnergy + wlDecoder.readDynamicEnergy +
                           senseAmp.readDynamicEnergy;
      writeDynamicEnergy +=
          precharger.writeDynamicEnergy + sramWriteDriver.writeDynamicEnergy +
          wlDecoder.writeDynamicEnergy + writeDynamicEnergyArray;
      leakage += precharger.leakage + sramWriteDriver.leakage + adder.leakage +
                 dff.leakage + wlDecoder.leakage + senseAmp.leakage;

    } else if (BNNparallelMode || XNORparallelMode) {
      precharger.CalculatePower(numOfColumnsMux, numOfRows * activityRowWrite);
      sramWriteDriver.CalculatePower(numOfRows * activityRowWrite);
      wlSwitchMatrix.CalculatePower(numOfColumnsMux,
                                    2 * numOfRows * activityRowWrite);
      if (isSarADC) {
        sarADC.CalculatePower(colResistance, 1);
      } else { // MLSA
        multilevelSenseAmp.CalculatePower(colResistance, 1);
        // Start From Here:
        multilevelSAEncoder.CalculatePower(numOfColumnsMux);
      }
      double readDynamicEnergyArray = 0; // Just BL charging
      double writeDynamicEnergyArray =
          capSRAMCell * tech->vdd * tech->vdd * 2 * numOfColumns *
          activityColWrite * numOfRows * activityRowWrite; // flip Q and Q_bar
      readDynamicEnergy += precharger.readDynamicEnergy +
                           wlSwitchMatrix.readDynamicEnergy +
                           sarADC.readDynamicEnergy + readDynamicEnergyArray +
                           multilevelSenseAmp.readDynamicEnergy +
                           multilevelSAEncoder.readDynamicEnergy;
      writeDynamicEnergy +=
          precharger.writeDynamicEnergy + sramWriteDriver.writeDynamicEnergy +
          wlSwitchMatrix.writeDynamicEnergy + writeDynamicEnergyArray;
      leakage += precharger.leakage + sramWriteDriver.leakage +
                 wlSwitchMatrix.leakage + multilevelSAEncoder.leakage;

    } else {
      throw runtime_error("operation Mode Error!");
    }
  } else { // RRAM,FeFET
    double capCol = heightArray * 0.2e-15 / 1e-6;
    if (conventionalSequential) {
      // similar to numReadCellPerOperationNeuro for SRAM
      int numReadCells = ceil((double)numOfColumns / numOfColumnsMux);
      int numWriteCells = numOfColumns;
      adder.CalculatePower(numOfColumnsMux * activityRowRead, numReadCells);
      dff.CalculatePower(numOfColumnsMux * numOfRows * activityRowRead,
                         numReadCells * (adder.numOfBits + 1));
      if (numOfReadPulses > 1) {
        // Need ShiftAdder to handle multi-bit input
        shiftAdder.CalculatePower(numOfColumnsMux);
      }
      double numRead = numOfRows * activityRowRead * numOfColumnsMux;
      double numWrite = 2 * numOfRows * activityRowWrite;
      wlDecoder.CalculatePower(numRead, numWrite);
      // eNVM and FeFET need Decoder Driver
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.CalculatePower(numRead, numWrite);
      } else {
        wlDecoderDriver.CalculatePower(numReadCells, numWriteCells, numRead,
                                       numWrite);
      }
      slSwitchMatrix.CalculatePower(0, numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculatePower(numOfColumnsMux);
        muxDecoder.CalculatePower(numOfColumnsMux, 1);
      }
      if (isSarADC) {
        sarADC.CalculatePower(colResistance, numOfRows * activityRowRead);
      } else { // MLSA
        multilevelSenseAmp.CalculatePower(colResistance,
                                          numOfRows * activityRowRead);
        if (avgWeightBit > 1) {
          multilevelSAEncoder.CalculatePower(numRead);
        }
      }
      // TODO: training condition
      double readDynamicEnergyArray =
          (capCol * param->readVoltage * param->readVoltage * numReadCells +
           capWL * tech->vdd * tech->vdd) *
          numRead;
      double writeDynamicEnergyArray = 0; // TODO: why? Different from SRAM.
      double muxReadEnergy =
          (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy) /
          numOfReadPulses;

      readDynamicEnergy +=
          adder.readDynamicEnergy + dff.readDynamicEnergy +
          shiftAdder.readDynamicEnergy + wlDecoder.readDynamicEnergy +
          wlNewDecoderDriver.readDynamicEnergy +
          wlDecoderDriver.readDynamicEnergy + sarADC.readDynamicEnergy +
          multilevelSenseAmp.readDynamicEnergy + muxReadEnergy +
          multilevelSAEncoder.readDynamicEnergy + readDynamicEnergyArray;

      writeDynamicEnergy +=
          wlDecoder.writeDynamicEnergy + wlNewDecoderDriver.writeDynamicEnergy +
          wlDecoderDriver.writeDynamicEnergy +
          slSwitchMatrix.writeDynamicEnergy + writeDynamicEnergyArray;

      leakage += adder.leakage + dff.leakage + shiftAdder.leakage +
                 wlDecoder.leakage + wlNewDecoderDriver.leakage +
                 wlDecoderDriver.leakage + slSwitchMatrix.leakage +
                 muxDecoder.leakage + multilevelSAEncoder.leakage;
    } else if (conventionalParallel) {
      double numRead = numOfColumnsMux;
      double numWrite = 2 * numOfRows * activityRowWrite;
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.CalculatePower(numRead, numWrite);
      } else {
        wlSwitchMatrix.CalculatePower(numRead, numWrite);
      }
      slSwitchMatrix.CalculatePower(0, numWrite);
      if (numOfColumnsMux > 1) {
        // TODO: numOfColumnsMux change to 16?
        mux.CalculatePower(numRead);
        muxDecoder.CalculatePower(numRead, 1);
      }
      if (isSarADC) {
        sarADC.CalculatePower(colResistance, 1);
      } else { // MLSA
        multilevelSenseAmp.CalculatePower(colResistance, 1);
        multilevelSAEncoder.CalculatePower(numOfColumnsMux);
      }
      if (numOfReadPulses > 1) {
        // TODO: numOfReadPulses change to 1?
        shiftAdder.CalculatePower(numOfColumnsMux);
      }
      // TODO: training condition
      int numReadCells = ceil((double)numOfColumns / numOfColumnsMux);
      double readDynamicEnergyArray =
          (capCol * param->readVoltage * param->readVoltage * numReadCells +
           capWL * tech->vdd * tech->vdd * numOfRows * activityRowRead) *
              numRead +
          ((1 / param->resistanceOn) + (1 / param->resistanceOff)) / 2 *
              numOfColumns * numOfRows * param->readPulseWidth *
              param->readPulseWidth * param->readVoltage;
      // cout << "tech->vdd = " << tech->vdd << endl;
      // cout << "param->resistanceOn = " << param->resistanceOn << endl;
      // cout << "param->resistanceOff = " << param->resistanceOff << endl;
      // cout << "param->readPulseWidth = " << param->readPulseWidth << endl;
      // cout << "param->readVoltage = " << param->readVoltage << endl;
      // cout << "readDynamicEnergyArray Part1 = " << capCol *
      // param->readVoltage * param->readVoltage * numReadCells +
      //      capWL * tech->vdd * tech->vdd * numOfRows * activityRowRead <<
      //      endl;
      // cout << "readDynamicEnergyArray Part11 = " << capCol *
      // param->readVoltage * param->readVoltage * numReadCells << endl; cout <<
      // "capCol = " << capCol << endl; cout << "param->readVoltage = " <<
      // param->readVoltage << endl; cout << "numReadCells = " << numReadCells
      // << endl;
      double writeDynamicEnergyArray = 0; // TODO: why? Different from SRAM.
      double muxReadEnergy =
          (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy) /
          numOfReadPulses;

      // TODO: why *0.05??
      readDynamicEnergy +=
          wlNewSwitchMatrix.readDynamicEnergy +
          wlSwitchMatrix.readDynamicEnergy + muxReadEnergy +
          sarADC.readDynamicEnergy + multilevelSenseAmp.readDynamicEnergy +
          multilevelSAEncoder.readDynamicEnergy + shiftAdder.readDynamicEnergy +
          readDynamicEnergyArray * 0.05;
      writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy +
                            wlSwitchMatrix.writeDynamicEnergy +
                            slSwitchMatrix.writeDynamicEnergy +
                            writeDynamicEnergyArray;
      leakage += wlSwitchMatrix.leakage + wlNewSwitchMatrix.leakage +
                 slSwitchMatrix.leakage + muxDecoder.leakage +
                 multilevelSAEncoder.leakage + shiftAdder.leakage;

      // cout << "****** readDynamicEnergy Calculate Begins ******" << endl;
      // cout << "readDynamicEnergy = " << readDynamicEnergy << endl;
      // cout << "wlNewSwitchMatrix.readDynamicEnergy = "
      //      << wlNewSwitchMatrix.readDynamicEnergy << endl;
      // cout << "wlSwitchMatrix.readDynamicEnergy = "
      //      << wlSwitchMatrix.readDynamicEnergy << endl;
      // cout << "muxReadEnergy = " << muxReadEnergy << endl;
      // cout << "mux.readDynamicEnergy = " << mux.readDynamicEnergy << endl;
      // cout << "muxDecoder.readDynamicEnergy = " << muxDecoder.readDynamicEnergy << endl;
      // cout << "sarADC.readDynamicEnergy = " << sarADC.readDynamicEnergy << endl;
      // cout << "multilevelSenseAmp.readDynamicEnergy = "
      //      << multilevelSenseAmp.readDynamicEnergy << endl;
      // cout << "multilevelSAEncoder.readDynamicEnergy = "
      //      << multilevelSAEncoder.readDynamicEnergy << endl;
      // cout << "shiftAdder.readDynamicEnergy = " << shiftAdder.readDynamicEnergy
      //      << endl;
      // cout << "readDynamicEnergyArray = " << readDynamicEnergyArray << endl;
      // cout << "****** readDynamicEnergy Calculate Ends ******" << endl;

      // cout << "****** writeDynamicEnergy Calculate Begins ******" << endl;
      // cout << "****** writeDynamicEnergy Calculate Ends ******" << endl;

      // cout << "****** leakage Calculate Begins ******" << endl;
      // cout << "leakage = " << leakage << endl;
      // cout << "wlSwitchMatrix.leakage = " << wlSwitchMatrix.leakage << endl;
      // cout << "wlNewSwitchMatrix.leakage = " << wlNewSwitchMatrix.leakage
      //      << endl;
      // cout << "slSwitchMatrix.leakage = " << slSwitchMatrix.leakage << endl;
      // cout << "muxDecoder.leakage = " << muxDecoder.leakage << endl;
      // cout << "multilevelSAEncoder.leakage = " << multilevelSAEncoder.leakage
      //      << endl;
      // cout << "shiftAdder.leakage = " << shiftAdder.leakage << endl;
      // cout << "***** leakage Calculate Ends *****" << endl;
    } else if (BNNsequentialMode || XNORsequentialMode) {
      double numRead = numOfColumnsMux * numOfRows * activityRowRead;
      double numWrite = 2 * numOfRows * activityRowWrite;
      int numReadCells = ceil((double)numOfColumns / numOfColumnsMux);
      int numWriteCells = numOfColumns;
      adder.CalculatePower(numRead, numReadCells);
      dff.CalculatePower(numRead, numReadCells * (adder.numOfBits + 1));
      wlDecoder.CalculatePower(numRead, numWrite);
      if (param->accessType == CMOS_access) {
        wlNewDecoderDriver.CalculatePower(numRead, numWrite);
      } else {
        wlDecoderDriver.CalculatePower(numReadCells, numWriteCells, numRead,
                                       numWrite);
      }
      slSwitchMatrix.CalculatePower(0, numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculatePower(numOfColumnsMux);
        muxDecoder.CalculatePower(numOfColumnsMux, 1);
      }
      rowCurrentSenseAmp.CalculatePower(colResistance,
                                        numOfRows * activityRowRead);

      double readDynamicEnergyArray =
          (capCol * param->readVoltage * param->readVoltage * numReadCells +
           capWL * tech->vdd * tech->vdd * numOfRows * activityRowRead) *
          numOfColumnsMux;
      double writeDynamicEnergyArray = 0;
      double muxReadEnergy =
          (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy) /
          numOfReadPulses;
      readDynamicEnergy +=
          adder.readDynamicEnergy + dff.readDynamicEnergy +
          wlDecoder.readDynamicEnergy + wlNewDecoderDriver.readDynamicEnergy +
          wlDecoderDriver.readDynamicEnergy + muxReadEnergy +
          readDynamicEnergyArray + rowCurrentSenseAmp.readDynamicEnergy;

      writeDynamicEnergy +=
          wlDecoder.writeDynamicEnergy + wlNewDecoderDriver.writeDynamicEnergy +
          wlDecoderDriver.writeDynamicEnergy +
          slSwitchMatrix.writeDynamicEnergy + writeDynamicEnergyArray;

      leakage += adder.leakage + dff.leakage + wlDecoder.leakage +
                 wlNewDecoderDriver.leakage + wlDecoderDriver.leakage +
                 slSwitchMatrix.leakage + muxDecoder.leakage;
    } else if (BNNparallelMode || XNORparallelMode) {
      double numRead = numOfColumnsMux;
      double numWrite = 2 * numOfRows * activityRowWrite;
      if (param->accessType == CMOS_access) {
        wlNewSwitchMatrix.CalculatePower(numRead, numWrite);
      } else {
        wlSwitchMatrix.CalculatePower(numRead, numWrite);
      }
      slSwitchMatrix.CalculatePower(0, numWrite);
      if (numOfColumnsMux > 1) {
        mux.CalculatePower(numRead);
        muxDecoder.CalculatePower(numRead, 1);
      }
      if (isSarADC) {
        sarADC.CalculatePower(colResistance, 1);
      } else { // MLSA
        multilevelSenseAmp.CalculatePower(colResistance, 1);
        multilevelSAEncoder.CalculatePower(numRead);
      }
      int numReadCells = ceil((double)numOfColumns / numOfColumnsMux);
      double readDynamicEnergyArray =
          (capCol * param->readVoltage * param->readVoltage * numReadCells +
           capWL * tech->vdd * tech->vdd * numOfRows * activityRowRead) *
          numOfColumnsMux;
      double writeDynamicEnergyArray = 0;
      double muxReadEnergy =
          (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy) /
          numOfReadPulses;
      readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy +
                           wlSwitchMatrix.readDynamicEnergy + muxReadEnergy +
                           readDynamicEnergyArray + sarADC.readDynamicEnergy +
                           multilevelSenseAmp.readDynamicEnergy +
                           multilevelSAEncoder.readDynamicEnergy;
      writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy +
                            wlSwitchMatrix.writeDynamicEnergy +
                            slSwitchMatrix.writeDynamicEnergy +
                            writeDynamicEnergyArray;
      leakage += wlNewSwitchMatrix.leakage + wlSwitchMatrix.leakage +
                 slSwitchMatrix.leakage + muxDecoder.leakage +
                 multilevelSAEncoder.leakage;
    } else {
      throw runtime_error("operation Mode Error!");
    }
  }
}

} // namespace CoMN
