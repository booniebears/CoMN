
/**
 * @file SarADC.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/SarADC.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
SarADC::SarADC(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void SarADC::Initialize(int _levelOutput, int _numOfADCs) {
  levelOutput = _levelOutput;
  numOfADCs = _numOfADCs;
  widthNMOS = MIN_NMOS_SIZE * tech->featureSize;
  widthPMOS = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;
}

void SarADC::CalculateArea(double _newHeight, double _newWidth,
                           AreaModify _option) {
  // cout << "********** SarADC::CalculateArea Begins **********" << endl;
  double areaUnit;
  if (param->user_defined) {
    areaUnit = param->ADC_area;
  } else {
    double hNmos, wNmos, hPmos, wPmos;
    CalculateGateArea(INV, 1, widthNMOS, 0,
                      tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNmos,
                      &wNmos);
    CalculateGateArea(INV, 1, 0, widthPMOS,
                      tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hPmos,
                      &wPmos);
    areaUnit = (hNmos * wNmos) * (269 + (log2(levelOutput) - 1) * 109) +
               (hPmos * wPmos) * (209 + (log2(levelOutput) - 1) * 73);
    // cout << "hNmos = " << hNmos << endl;
    // cout << "wNmos = " << wNmos << endl;
    // cout << "hPmos = " << hPmos << endl;
    // cout << "wPmos = " << wPmos << endl;
  }
  area = numOfADCs * areaUnit;
  if (_newWidth && _option == NONE) {
    width = _newWidth;
    height = area / width;
  } else if (_newHeight && _option == NONE) {
    height = _newHeight;
    width = area / height;
  }
  // cout << "numOfADCs = " << numOfADCs << endl;
  // cout << "areaUnit = " << areaUnit << endl;
  // cout << "width = " << width << endl;
  // cout << "height = " << height << endl;
  // cout << "********** SarADC::CalculateArea Ends **********" << endl;
}

void SarADC::CalculateLatency(double numRead) {
  // numRead = numOfColumnsMux.
  // cout << "********** SarADC::CalculateLatency Begins **********" << endl;
  if (param->user_defined) {
    readLatency = param->ADC_delay * numRead;
  } else {
    readLatency = (log2(levelOutput) + 1) / param->clkFreq * numRead;
  }
  // cout << "levelOutput = " << levelOutput << endl;
  // cout << "param->clkFreq = " << param->clkFreq << endl;
  // cout << "numRead = " << numRead << endl;
  // cout << "********** SarADC::CalculateLatency Ends **********" << endl;
}

double SarADC::GetColumnPower(double resCol) {
  //
  double Column_Power = 0;
  double Column_Energy = 0;
  // in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread
  // (different from 0.5V) we should modify the equivalent resCol
  if (param->user_defined == 0) {
    resCol *= 0.5 / param->readVoltage;
    if ((double)1 / resCol == 0) {
      Column_Power = 1e-6;
    } else if (resCol == 0) {
      Column_Power = 0;
    } else {
      if (param->deviceRoadmap == HP) { // HP
        if (param->technode == 130) {
          Column_Power = (6.4806 * log2(levelOutput) + 49.047) * 1e-6;
          Column_Power += 0.207452 * exp(-2.367 * log10(resCol));
        } else if (param->technode == 90) {
          Column_Power = (4.3474 * log2(levelOutput) + 31.782) * 1e-6;
          Column_Power += 0.164900 * exp(-2.345 * log10(resCol));
        } else if (param->technode == 65) {
          Column_Power = (2.9503 * log2(levelOutput) + 22.047) * 1e-6;
          Column_Power += 0.128483 * exp(-2.321 * log10(resCol));
        } else if (param->technode == 45) {
          Column_Power = (2.1843 * log2(levelOutput) + 11.931) * 1e-6;
          Column_Power += 0.097754 * exp(-2.296 * log10(resCol));
        } else if (param->technode == 32) {
          Column_Power = (1.0157 * log2(levelOutput) + 7.6286) * 1e-6;
          Column_Power += 0.083709 * exp(-2.313 * log10(resCol));
        } else if (param->technode == 22) {
          Column_Power = (0.7213 * log2(levelOutput) + 3.3041) * 1e-6;
          Column_Power += 0.084273 * exp(-2.311 * log10(resCol));
        } else if (param->technode == 14) {
          Column_Power = (0.4710 * log2(levelOutput) + 1.9529) * 1e-6;
          Column_Power += 0.060584 * exp(-2.311 * log10(resCol));
        } else if (param->technode == 10) {
          Column_Power = (0.3076 * log2(levelOutput) + 1.1543) * 1e-6;
          Column_Power += 0.049418 * exp(-2.311 * log10(resCol));
        } else { // 7nm
          Column_Power = (0.2008 * log2(levelOutput) + 0.6823) * 1e-6;
          Column_Power += 0.040310 * exp(-2.311 * log10(resCol));
        }
      } else { // LP
        if (param->technode == 130) {
          Column_Power = (8.4483 * log2(levelOutput) + 65.243) * 1e-6;
          Column_Power += 0.169380 * exp(-2.303 * log10(resCol));
        } else if (param->technode == 90) {
          Column_Power = (5.9869 * log2(levelOutput) + 37.462) * 1e-6;
          Column_Power += 0.144323 * exp(-2.303 * log10(resCol));
        } else if (param->technode == 65) {
          Column_Power = (3.7506 * log2(levelOutput) + 25.844) * 1e-6;
          Column_Power += 0.121272 * exp(-2.303 * log10(resCol));
        } else if (param->technode == 45) {
          Column_Power = (2.1691 * log2(levelOutput) + 16.693) * 1e-6;
          Column_Power += 0.100225 * exp(-2.303 * log10(resCol));
        } else if (param->technode == 32) {
          Column_Power = (1.1294 * log2(levelOutput) + 8.8998) * 1e-6;
          Column_Power += 0.079449 * exp(-2.297 * log10(resCol));
        } else if (param->technode == 22) {
          Column_Power = (0.538 * log2(levelOutput) + 4.3753) * 1e-6;
          Column_Power += 0.072341 * exp(-2.303 * log10(resCol));
        } else if (param->technode == 14) {
          Column_Power = (0.3132 * log2(levelOutput) + 2.5681) * 1e-6;
          Column_Power += 0.061085 * exp(-2.303 * log10(resCol));
        } else if (param->technode == 10) {
          Column_Power = (0.1823 * log2(levelOutput) + 1.5073) * 1e-6;
          Column_Power += 0.051580 * exp(-2.303 * log10(resCol));
        } else { // 7nm
          Column_Power = (0.1061 * log2(levelOutput) + 0.8847) * 1e-6;
          Column_Power += 0.043555 * exp(-2.303 * log10(resCol));
        }
      }
    }
    Column_Power *= (1 + 1.3e-3 * (param->temp - 300));
    Column_Energy = Column_Power * (log2(levelOutput) + 1) * 1 / param->clkFreq;
  } else {
    Column_Energy = param->ADC_power * param->ADC_delay;
  }
  return Column_Energy;
}

void SarADC::CalculatePower(vector<double> &colResistance, double numRead) {
  // No leakage in SarADC.
  for (auto res : colResistance) {
    double E_col = GetColumnPower(res);
    readDynamicEnergy += E_col;
  }
  readDynamicEnergy *= numRead;
}

} // namespace CoMN
