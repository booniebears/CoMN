/**
 * @file MuxDecoder.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/MuxDecoder.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
MuxDecoder::MuxDecoder(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void MuxDecoder::Initialize(DecoderMode _mode, int _inputBits, bool _isMux,
                            bool _isParallel) {
  mode = _mode;
  inputBits = _inputBits;
  isMux = _isMux;
  isParallel = _isParallel;

  if (isParallel) { // increase isMux Decoder by 8 times
    // Use 2-bit predecoding
    // INV
    widthInvN = 8 * MIN_NMOS_SIZE * tech->featureSize;
    widthInvP = 8 * tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
    numInv = inputBits; // The INV at outpur driver stage does not count here

    // NAND2
    widthNandN = 8 * 2 * MIN_NMOS_SIZE * tech->featureSize;
    widthNandP = 8 * tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
    numNand = 4 * (int)(floor(inputBits / 2));

    // NOR (ceil(N/2) inputs)
    widthNorN = 8 * MIN_NMOS_SIZE * tech->featureSize;
    widthNorP = 8 * (int)ceil((double)inputBits / 2) * tech->pnSizeRatio *
                MIN_NMOS_SIZE * tech->featureSize;
    if (inputBits > 2)
      numNor = pow(2, inputBits);
    else
      numNor = 0;

    // Number of M3 for connection between NAND2 and NOR stages (inputBits > 2)
    if (inputBits > 2)
      numMetalConnection = numNand + (inputBits % 2) * 2;
    else
      numMetalConnection = 0;
    // Output driver INV
    widthDriverInvN = 8 * 3 * MIN_NMOS_SIZE * tech->featureSize;
    widthDriverInvP =
        8 * 3 * tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
  }

  else {
    // Use 2-bit predecoding
    // INV
    widthInvN = MIN_NMOS_SIZE * tech->featureSize;
    widthInvP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
    numInv = inputBits; // The INV at outpur driver stage does not count here

    // NAND2
    widthNandN = 2 * MIN_NMOS_SIZE * tech->featureSize;
    widthNandP = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
    numNand = 4 * (int)(floor(inputBits / 2));

    // NOR (ceil(N/2) inputs)
    widthNorN = MIN_NMOS_SIZE * tech->featureSize;
    widthNorP = (int)ceil((double)inputBits / 2) * tech->pnSizeRatio *
                MIN_NMOS_SIZE * tech->featureSize;
    if (inputBits > 2) {
      numNor = pow(2, inputBits);
      // Number of M3 for connection between NAND2 and NOR stages (inputBits >
      // 2)
      numMetalConnection = numNand + (inputBits % 2) * 2;
      // cout << "numNand = " << numNand << endl;
      // cout << "inputBits = " << inputBits << endl;
    } else {
      numNor = 0;
      numMetalConnection = 0;
    }

    // Output driver INV
    widthDriverInvN = 3 * MIN_NMOS_SIZE * tech->featureSize;
    widthDriverInvP = 3 * tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
  }
}

void MuxDecoder::CalculateArea(double _newHeight, double _newWidth,
                               AreaModify _option) {
  // cout << "********** MuxDecoder::CalculateArea Begins **********" << endl;
  double hInv, wInv, hNand, wNand, hNor, wNor, hDriverInv, wDriverInv;
  // INV
  CalculateGateArea(INV, 1, widthInvN, widthInvP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv,
                    &wInv);
  // NAND2
  CalculateGateArea(NAND, 2, widthNandN, widthNandP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand,
                    &wNand);
  // NOR (ceil(N/2) inputs)
  CalculateGateArea(NOR, (int)ceil((double)inputBits / 2), widthNorN, widthNorP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNor,
                    &wNor);
  // Output Driver INV
  CalculateGateArea(INV, 1, widthDriverInvN, widthDriverInvP,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech,
                    &hDriverInv, &wDriverInv);
  if (mode == REGULAR_ROW) {
    if (_newHeight && _option == NONE) {
      if (hInv > _newHeight || hNand > _newHeight || hNor > _newHeight) {
        throw runtime_error("(MuxDecoder.cpp)[Error]: Gate Height greater than "
                            "assigned Height!!!");
      }
      // Inv
      int numOfInvsPerCol = _newHeight / hInv;
      if (numOfInvsPerCol > numInv) {
        numOfInvsPerCol = numInv;
      }
      int numOfColumnsInv = ceil((double)numInv / numOfInvsPerCol);
      // NAND
      int numOfNANDsPerCol = _newHeight / hNand;
      if (numOfNANDsPerCol > numNand) {
        numOfNANDsPerCol = numNand;
      }
      int numOfColumnsNAND = ceil((double)numNand / numOfNANDsPerCol);
      // NOR
      int numOfNORsPerCol = _newHeight / hNor;
      if (numOfNORsPerCol > numNor) {
        numOfNORsPerCol = numNor;
      }
      int numOfColumnsNOR = ceil((double)numNor / numOfNORsPerCol);

      width = wInv * numOfColumnsInv + wNand * numOfColumnsNAND +
              M3_PITCH * numMetalConnection * tech->featureSize +
              wNor * numOfColumnsNOR; // TODO: M3_PITCH??
      height = _newHeight;
      if (isMux) { // Mux enable circuit (NAND + INV) + INV
        width += (wNand + wInv * 2) * numOfColumnsNOR;
      } else { // REGULAR: 2 INV as output driver
        width += (wDriverInv * 2) * numOfColumnsNOR;
      }
    } else {
      width = wInv + wNand + M3_PITCH * numMetalConnection * tech->featureSize +
              wNor;
      height = max(hNor * numNor, hNand * numNand);
      if (isMux) { // Mux enable circuit (NAND + INV) + INV
        width += wNand + wInv * 2;
      } else { // REGULAR: 2 INV as output driver
        width += wDriverInv * 2;
      }
    }
    // cout << "width = " << width << endl;
    // cout << "height = " << height << endl;
    // cout << "hNor = " << hNor << endl;
    // cout << "hNand = " << hNand << endl;
    // cout << "numMetalConnection = " << numMetalConnection << endl;
  } else { // REGULAR_COL
    if (_newWidth && _option == NONE) {
      if (wInv > _newWidth || wNand > _newWidth || wNor > _newWidth) {
        throw runtime_error("(MuxDecoder.cpp)[Error]: Gate Width greater than "
                            "assigned Width!!!");
      }
      // Inv
      int numOfInvsPerRow = _newWidth / wInv;
      if (numOfInvsPerRow > numInv) {
        numOfInvsPerRow = numInv;
      }
      int numOfRowsInv = ceil((double)numInv / numOfInvsPerRow);
      // NAND
      int numOfNANDsPerRow = _newWidth / wNand;
      if (numOfNANDsPerRow > numNand) {
        numOfNANDsPerRow = numNand;
      }
      int numOfRowsNAND = ceil((double)numNand / numOfNANDsPerRow);
      // NOR
      int numOfNORsPerRow = _newWidth / wNor;
      if (numOfNORsPerRow > numNor) {
        numOfNORsPerRow = numNor;
      }
      int numOfRowsNOR = ceil((double)numNor / numOfNORsPerRow);
      width = _newWidth;
      height = hInv * numOfRowsInv + hNand * numOfRowsNAND +
               M2_PITCH * numMetalConnection * tech->featureSize +
               hNor * numOfRowsNOR; // TODO: M2_PITCH??
      if (isMux) {                  // Mux enable circuit (NAND + INV) + INV
        height += (hNand + hInv * 2) * numOfRowsNOR;
      } else { // REGULAR: 2 INV as output driver
        height += (hDriverInv * 2) * numOfRowsNOR;
      }
    } else {
      height = hInv + hNand +
               M2_PITCH * numMetalConnection * tech->featureSize + hNor;
      width = max(wNor * numNor, wNand * numNand);
      if (isMux) { // Mux enable circuit (NAND + INV) + INV
        height += hNand + hInv * 2;
      } else { // REGULAR: 2 INV as output driver
        height += hDriverInv * 2;
      }
    }
  }
  area = height * width;
  // Capacitance
  // INV
  CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech,
                           &capInvInput, &capInvOutput);
  // NAND2
  if (numNand) {
    CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech,
                             &capNandInput, &capNandOutput);
  } else {
    capNandInput = capNandOutput = 0;
  }
  // NOR (ceil(N/2) inputs)
  if (numNor) {
    CalculateGateCapacitance(NOR, (int)ceil((double)inputBits / 2), widthNorN,
                             widthNorP, hNor, tech, &capNorInput,
                             &capNorOutput);
  } else {
    capNorInput = capNorOutput = 0;
  }
  // Output Driver INV
  CalculateGateCapacitance(INV, 1, widthDriverInvN, widthDriverInvP, hDriverInv,
                           tech, &capDriverInvInput, &capDriverInvOutput);
  // cout << "********** MuxDecoder::CalculateArea Ends **********" << endl;
}

void MuxDecoder::CalculateLatency(double _rampInput, double _capLoad1,
                                  double _capLoad2, double numRead,
                                  double numWrite) {
  double resPullDown, resPullUp;
  double tr;   /* time constant */
  double gm;   /* transconductance */
  double beta; /* for horowitz calculation */
  double rampInvOutput = INF_RAMP;
  double rampNandOutput = INF_RAMP;
  double rampNorOutput = INF_RAMP;

  // INV
  // doesn't matter pullup/pulldown?
  resPullDown = CalculateOnResistance(widthInvN, NMOS, param->temp, tech);
  if (numNand)
    // one address line connects to 2 NAND inputs
    tr = resPullDown * (capInvOutput + capNandInput * 2);
  else
    tr = resPullDown * (capInvOutput + _capLoad1);
  gm = CalculateTransconductance(widthInvN, NMOS, tech);
  beta = 1 / (resPullDown * gm);
  readLatency += horowitz(tr, beta, _rampInput, &rampInvOutput);
  writeLatency += horowitz(tr, beta, _rampInput, &rampInvOutput);

  if (!numNand)
    rampOutput = rampInvOutput;

  // NAND2
  if (numNand) {
    resPullDown =
        CalculateOnResistance(widthNandN, NMOS, param->temp, tech) * 2;
    if (numNor)
      tr = resPullDown * (capNandOutput + capNorInput * numNor / 4);
    else
      tr = resPullDown * (capNandOutput + _capLoad1);
    gm = CalculateTransconductance(widthNandN, NMOS, tech);
    beta = 1 / (resPullDown * gm);
    readLatency += horowitz(tr, beta, rampInvOutput, &rampNandOutput);
    writeLatency += horowitz(tr, beta, rampInvOutput, &rampNandOutput);
    if (!numNor)
      rampOutput = rampNandOutput;
  }

  // NOR (ceil(N/2) inputs)
  if (numNor) {
    resPullUp = CalculateOnResistance(widthNorP, PMOS, param->temp, tech) * 2;
    if (isMux)
      tr = resPullUp * (capNorOutput + capNandInput);
    else
      tr = resPullUp * (capNorOutput + capInvInput);
    gm = CalculateTransconductance(widthNorP, PMOS, tech);
    beta = 1 / (resPullUp * gm);
    readLatency += horowitz(tr, beta, rampNandOutput, &rampNorOutput);
    writeLatency += horowitz(tr, beta, rampNandOutput, &rampNorOutput);
    rampOutput = rampNorOutput;
  }

  // Output driver or Mux enable circuit
  if (isMux) { // Mux enable circuit (NAND + INV) + INV
    // 1st NAND
    resPullDown = CalculateOnResistance(widthNandN, NMOS, param->temp, tech);
    tr = resPullDown * (capNandOutput + capInvInput);
    gm = CalculateTransconductance(widthNandN, NMOS, tech);
    beta = 1 / (resPullDown * gm);
    readLatency += horowitz(tr, beta, rampNorOutput, &rampNandOutput);
    writeLatency += horowitz(tr, beta, rampNorOutput, &rampNandOutput);
    // 2nd INV
    resPullUp = CalculateOnResistance(widthInvP, PMOS, param->temp, tech);
    tr = resPullUp * (capInvOutput + capInvInput + _capLoad1);
    gm = CalculateTransconductance(widthInvP, PMOS, tech);
    beta = 1 / (resPullUp * gm);
    readLatency += horowitz(tr, beta, rampNandOutput, &rampInvOutput);
    writeLatency += horowitz(tr, beta, rampNandOutput, &rampInvOutput);
    // 3rd INV
    resPullDown = CalculateOnResistance(widthInvN, NMOS, param->temp, tech);
    tr = resPullDown * (capInvOutput + _capLoad2);
    gm = CalculateTransconductance(widthInvN, NMOS, tech);
    beta = 1 / (resPullDown * gm);
    readLatency += horowitz(tr, beta, rampInvOutput, &rampOutput);
    writeLatency += horowitz(tr, beta, rampInvOutput, &rampOutput);
    rampOutput = rampInvOutput;
  } else { // REGULAR: 2 INV as output driver
    // 1st INV
    resPullDown =
        CalculateOnResistance(widthDriverInvN, NMOS, param->temp, tech);
    tr = resPullDown * (capDriverInvOutput + capDriverInvInput);
    gm = CalculateTransconductance(widthDriverInvN, NMOS, tech);
    beta = 1 / (resPullDown * gm);
    readLatency += horowitz(tr, beta, rampNorOutput, &rampInvOutput);
    writeLatency += horowitz(tr, beta, rampNorOutput, &rampInvOutput);
    // 2nd INV
    resPullUp = CalculateOnResistance(widthDriverInvP, PMOS, param->temp, tech);
    tr = resPullUp * (capDriverInvOutput + _capLoad1);
    gm = CalculateTransconductance(widthDriverInvP, PMOS, tech);
    beta = 1 / (resPullUp * gm);
    readLatency += horowitz(tr, beta, rampInvOutput, &rampOutput);
    writeLatency += horowitz(tr, beta, rampInvOutput, &rampOutput);
    rampOutput = rampInvOutput;
  }
  readLatency *= numRead;
  writeLatency *= numWrite;
}

void MuxDecoder::CalculatePower(double numRead, double numWrite) {
  // cout << "********** MuxDecoder::CalculatePower Begins **********" << endl;
  leakage +=
      CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
      tech->vdd * numInv;
  leakage +=
      CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, param->temp, tech) *
      tech->vdd * numNand;
  leakage += CalculateGateLeakage(NOR, ceil((double)inputBits / 2), widthNorN,
                                  widthNorP, param->temp, tech) *
             tech->vdd * numNor;

  // Output driver or Mux enable circuit
  if (isMux) {
    leakage += CalculateGateLeakage(NAND, 2, widthNandN, widthNandP,
                                    param->temp, tech) *
               tech->vdd * numNor;
    leakage +=
        CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param->temp, tech) *
        tech->vdd * 2 * numNor;
  } else {
    leakage += CalculateGateLeakage(INV, 1, widthDriverInvN, widthDriverInvP,
                                    param->temp, tech) *
               tech->vdd * 2 * numNor;
  }

  // Read dynamic energy for both memory and neuro modes (rough calculation
  // assuming all addr from 0 to 1) INV
  readDynamicEnergy += (capInvInput + capNandInput * 2) * tech->vdd *
                       tech->vdd * (int)floor(inputBits / 2) * 2;
  readDynamicEnergy +=
      (capInvInput + capNorInput * numNor / 2) * tech->vdd * tech->vdd *
      (inputBits - (int)floor(inputBits / 2) * 2); // If inputBits is odd number
  // NAND2
  readDynamicEnergy +=
      (capNandOutput + capNorInput * numNor / 4) * tech->vdd * tech->vdd *
      numNand / 4; // every (NAND * 4) group has one NAND output activated

  // INV
  writeDynamicEnergy += (capInvInput + capNandInput * 2) * tech->vdd *
                        tech->vdd * (int)floor(inputBits / 2) * 2;
  writeDynamicEnergy +=
      (capInvInput + capNorInput * numNor / 2) * tech->vdd * tech->vdd *
      (inputBits - (int)floor(inputBits / 2) * 2); // If inputBits is odd number
  // NAND2
  writeDynamicEnergy +=
      (capNandOutput + capNorInput * numNor / 4) * tech->vdd * tech->vdd *
      numNand / 4; // every (NAND * 4) group has one NAND output activated

  // NOR (ceil(N/2) inputs)
  if (isMux) {
    // one NOR output activated
    readDynamicEnergy += (capNorOutput + capNandInput) * tech->vdd * tech->vdd;
    writeDynamicEnergy += (capNorOutput + capNandInput) * tech->vdd * tech->vdd;
  } else {
    // one NOR output activated
    readDynamicEnergy += (capNorOutput + capInvInput) * tech->vdd * tech->vdd;
    writeDynamicEnergy += (capNorOutput + capInvInput) * tech->vdd * tech->vdd;
  }

  // Output driver or Mux enable circuit
  if (isMux) {
    readDynamicEnergy += (capNandOutput + capInvInput) * tech->vdd * tech->vdd;
    readDynamicEnergy += (capInvOutput + capInvInput) * tech->vdd * tech->vdd;
    readDynamicEnergy += capInvOutput * tech->vdd * tech->vdd;

    writeDynamicEnergy += (capNandOutput + capInvInput) * tech->vdd * tech->vdd;
    writeDynamicEnergy += (capInvOutput + capInvInput) * tech->vdd * tech->vdd;
    writeDynamicEnergy += capInvOutput * tech->vdd * tech->vdd;
  } else {
    readDynamicEnergy +=
        (capDriverInvInput + capDriverInvOutput) * tech->vdd * tech->vdd * 2;
    writeDynamicEnergy +=
        (capDriverInvInput + capDriverInvOutput) * tech->vdd * tech->vdd * 2;
  }
  readDynamicEnergy *= numRead;
  writeDynamicEnergy *= numWrite;
  // cout << "********** MuxDecoder::CalculatePower Ends **********" << endl;
}

} // namespace CoMN
