/**
 * @file CurrentSenseAmp.cpp
 * @author booniebears
 * @brief
 * @date 2023-10-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "../include/CurrentSenseAmp.h"
#include "../include/general/Constant.h"
#include "../include/general/Formula.h"

namespace CoMN {
CurrentSenseAmp::CurrentSenseAmp(Param *_param, Technology *_technology)
    : param(_param), tech(_technology), BasicUnit() {}

void CurrentSenseAmp::Initialize(int _numOfColumns) {
  numOfColumns = _numOfColumns;

  widthNMOS = MIN_NMOS_SIZE * tech->featureSize;
  widthPMOS = tech->pnSizeRatio * MIN_NMOS_SIZE * tech->featureSize;

  double R_start = (double)1 / param->maxConductance;
  double R_index =
      (double)1 / param->minConductance - (double)1 / param->maxConductance;
  Rref = R_start + (double)R_index / 2;
}

void CurrentSenseAmp::CalculateArea(double _newHeight, double _newWidth,
                                    AreaModify _option) {
  double hNmos, wNmos, hPmos, wPmos;
  // TODO: Obviously wrong implementation in the original code???
  CalculateGateArea(INV, 1, widthNMOS, 0,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNmos,
                    &wNmos);
  CalculateGateArea(INV, 1, 0, widthPMOS,
                    tech->featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hPmos,
                    &wPmos);
  double areaUnit = (hNmos * wNmos) * 48 + (hPmos * wPmos) * 40;

  if (_newWidth && _option == NONE) {
    // area = HEIGHT_WIDTH_RATIO_LIMIT * x^2
    double x = sqrt(areaUnit / HEIGHT_WIDTH_RATIO_LIMIT);
    if (_newWidth > x) // Limit W/H <= HEIGHT_WIDTH_RATIO_LIMIT
      _newWidth = x;
    area = areaUnit * numOfColumns;
    width = _newWidth;
    height = area / _newWidth;
  } else {
    throw runtime_error("(CurrentSenseAmp.cpp)[Error]: _newWidth not given!!");
  }
}

double CurrentSenseAmp::GetColumnLatency(double resCol) {
  double Column_Latency = 0;
  double up_bound = 3, mid_bound = 1.1, low_bound = 0.9;
  double T_max = 0;
  // in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread
  // (different from 0.5V) we should modify the equivalent resCol
  resCol *= 0.5 / param->readVoltage;
  if (((double)1 / resCol == 0) || (resCol == 0)) {
    Column_Latency = 0;
  } else {
    if (param->deviceRoadmap == HP) { // HP
      Column_Latency = 1e-9;
    } else { // LP
      if (param->technode == 130) {
        T_max = (0.2679 * log(resCol / 1000) + 0.0478) *
                1e-9; // T_max = (0.2679*log(R_BL/1000)+0.0478)*10^-9;

        double ratio = Rref / resCol;
        double T = 0;
        if (ratio >= 20 || ratio <= 0.05) {
          T = 1e-9;
        } else {
          if (ratio <= low_bound) {
            T = T_max * (3.915 * pow(ratio, 3) - 5.3996 * pow(ratio, 2) +
                         2.4653 * ratio +
                         0.3856); // y = 3.915*x^3-5.3996*x^2+2.4653*x+0.3856;
          } else if (mid_bound <= ratio <= up_bound) {
            T = T_max *
                (0.0004 * pow(ratio, 4) - 0.0087 * pow(ratio, 3) +
                 0.0742 * pow(ratio, 2) - 0.2725 * ratio +
                 1.2211); // y =
                          // 0.0004*x^4-0.0087*x^3+0.0742*x^2-0.2725*x+1.2211;
          } else if (ratio > up_bound) {
            T = T_max * (0.0004 * pow(ratio, 4) - 0.0087 * pow(ratio, 3) +
                         0.0742 * pow(ratio, 2) - 0.2725 * ratio + 1.2211);
          } else {
            T = T_max;
          }
        }
        Column_Latency = max(Column_Latency, T);

      } else if (param->technode == 90) {
        T_max = (0.0586 * log(resCol / 1000) + 1.41) *
                1e-9; // T_max = (0.0586*log(R_BL/1000)+1.41)*10^-9;

        double ratio = Rref / resCol;
        double T = 0;
        if (ratio >= 20 || ratio <= 0.05) {
          T = 1e-9;
        } else {
          if (ratio <= low_bound) {
            T = T_max * (3.726 * pow(ratio, 3) - 5.651 * pow(ratio, 2) +
                         2.8249 * ratio +
                         0.3574); // y = 3.726*x^3-5.651*x^2+2.8249*x+0.3574;
          } else if (mid_bound <= ratio <= up_bound) {
            T = T_max *
                (0.0000008 * pow(ratio, 4) - 0.00007 * pow(ratio, 3) +
                 0.0017 * pow(ratio, 2) - 0.0188 * ratio +
                 0.9835); // y =
                          // 0.0000008*x^4-0.00007*x^3+0.0017*x^2-0.0188*x+0.9835;
          } else if (ratio > up_bound) {
            T = T_max * (0.0000008 * pow(ratio, 4) - 0.00007 * pow(ratio, 3) +
                         0.0017 * pow(ratio, 2) - 0.0188 * ratio + 0.9835);
          } else {
            T = T_max;
          }
        }
        Column_Latency = max(Column_Latency, T);

      } else if (param->technode == 65) {
        T_max = (0.1239 * log(resCol / 1000) + 0.6642) *
                1e-9; // T_max = (0.1239*log(R_BL/1000)+0.6642)*10^-9;

        double ratio = Rref / resCol;
        double T = 0;
        if (ratio >= 20 || ratio <= 0.05) {
          T = 1e-9;
        } else {
          if (ratio <= low_bound) {
            T = T_max * (1.3899 * pow(ratio, 3) - 2.6913 * pow(ratio, 2) +
                         2.0483 * ratio +
                         0.3202); // y = 1.3899*x^3-2.6913*x^2+2.0483*x+0.3202;
          } else if (mid_bound <= ratio <= up_bound) {
            T = T_max *
                (0.0036 * pow(ratio, 4) - 0.0363 * pow(ratio, 3) +
                 0.1043 * pow(ratio, 2) - 0.0346 * ratio +
                 1.0512); // y =
                          // 0.0036*x^4-0.0363*x^3+0.1043*x^2-0.0346*x+1.0512;
          } else if (ratio > up_bound) {
            T = T_max * (0.0036 * pow(ratio, 4) - 0.0363 * pow(ratio, 3) +
                         0.1043 * pow(ratio, 2) - 0.0346 * ratio + 1.0512);
          } else {
            T = T_max;
          }
        }
        Column_Latency = max(Column_Latency, T);

      } else if (param->technode == 45 || param->technode == 32) {
        T_max = (0.0714 * log(resCol / 1000) + 0.7651) *
                1e-9; // T_max = (0.0714*log(R_BL/1000)+0.7651)*10^-9;

        double ratio = Rref / resCol;
        double T = 0;
        if (ratio >= 20 || ratio <= 0.05) {
          T = 1e-9;
        } else {
          if (ratio <= low_bound) {
            T = T_max * (3.7949 * pow(ratio, 3) - 5.6685 * pow(ratio, 2) +
                         2.6492 * ratio +
                         0.4807); // y = 3.7949*x^3-5.6685*x^2+2.6492*x+0.4807
          } else if (mid_bound <= ratio <= up_bound) {
            T = T_max *
                (0.000001 * pow(ratio, 4) - 0.00006 * pow(ratio, 3) +
                 0.0001 * pow(ratio, 2) - 0.0171 * ratio +
                 1.0057); // 0.000001*x^4-0.00006*x^3+0.0001*x^2-0.0171*x+1.0057;
          } else if (ratio > up_bound) {
            T = T_max * (0.000001 * pow(ratio, 4) - 0.00006 * pow(ratio, 3) +
                         0.0001 * pow(ratio, 2) - 0.0171 * ratio + 1.0057);
          } else {
            T = T_max;
          }
        }
        Column_Latency = max(Column_Latency, T);

      } else { // technode below and equal to 22nm
        Column_Latency = 1e-9;
      }
    }
  }
  return Column_Latency;
}

void CurrentSenseAmp::CalculateLatency(vector<double> &colResistance,
                                       double numRead) {
  double LatencyCol = 0;

  for (auto res : colResistance) {
    double T_col = GetColumnLatency(res);
    LatencyCol = max(LatencyCol, T_col);
    if (LatencyCol < 5e-10) {
      LatencyCol = 5e-10;
    } else if (LatencyCol > 50e-9) {
      LatencyCol = 50e-9;
    }
  }
  readLatency = LatencyCol * numRead;
}

double CurrentSenseAmp::GetColumnPower(double resCol) {
  double Column_Power = 0;
  // in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread
  // (different from 0.5V) we should modify the equivalent resCol
  resCol *= 0.5 / param->readVoltage;
  if ((double)1 / resCol == 0) {
    Column_Power = 1e-6;
  } else if (resCol == 0) {
    Column_Power = 0;
  } else {
    if (param->deviceRoadmap == HP) { // HP
      if (param->technode == 130) {
        Column_Power = 19.898 * 1e-6;
        Column_Power += 0.207452 * exp(-2.367 * log10(resCol));
      } else if (param->technode == 90) {
        Column_Power = 13.09 * 1e-6;
        Column_Power += 0.164900 * exp(-2.345 * log10(resCol));
      } else if (param->technode == 65) {
        Column_Power = 9.9579 * 1e-6;
        Column_Power += 0.128483 * exp(-2.321 * log10(resCol));
      } else if (param->technode == 45) {
        Column_Power = 7.7017 * 1e-6;
        Column_Power += 0.097754 * exp(-2.296 * log10(resCol));
      } else if (param->technode == 32) {
        Column_Power = 3.9648 * 1e-6;
        Column_Power += 0.083709 * exp(-2.313 * log10(resCol));
      } else if (param->technode == 22) {
        Column_Power = 1.8939 * 1e-6;
        Column_Power += 0.084273 * exp(-2.311 * log10(resCol));
      } else if (param->technode == 14) {
        Column_Power = 1.2 * 1e-6;
        Column_Power += 0.060584 * exp(-2.311 * log10(resCol));
      } else if (param->technode == 10) {
        Column_Power = 0.8 * 1e-6;
        Column_Power += 0.049418 * exp(-2.311 * log10(resCol));
      } else { // 7nm
        Column_Power = 0.5 * 1e-6;
        Column_Power += 0.040310 * exp(-2.311 * log10(resCol));
      }
    } else { // LP
      if (param->technode == 130) {
        Column_Power = 18.09 * 1e-6;
        Column_Power += 0.169380 * exp(-2.303 * log10(resCol));
      } else if (param->technode == 90) {
        Column_Power = 12.612 * 1e-6;
        Column_Power += 0.144323 * exp(-2.303 * log10(resCol));
      } else if (param->technode == 65) {
        Column_Power = 8.4147 * 1e-6;
        Column_Power += 0.121272 * exp(-2.303 * log10(resCol));
      } else if (param->technode == 45) {
        Column_Power = 6.3162 * 1e-6;
        Column_Power += 0.100225 * exp(-2.303 * log10(resCol));
      } else if (param->technode == 32) {
        Column_Power = 3.0875 * 1e-6;
        Column_Power += 0.079449 * exp(-2.297 * log10(resCol));
      } else if (param->technode == 22) {
        Column_Power = 1.7 * 1e-6;
        Column_Power += 0.072341 * exp(-2.303 * log10(resCol));
      } else if (param->technode == 14) {
        Column_Power = 1.0 * 1e-6;
        Column_Power += 0.061085 * exp(-2.303 * log10(resCol));
      } else if (param->technode == 10) {
        Column_Power = 0.55 * 1e-6;
        Column_Power += 0.051580 * exp(-2.303 * log10(resCol));
      } else { // 7nm
        Column_Power = 0.35 * 1e-6;
        Column_Power += 0.043555 * exp(-2.303 * log10(resCol));
      }
    }
  }
  return Column_Power;
}

void CurrentSenseAmp::CalculatePower(vector<double> &colResistance,
                                     double numRead) {
  //
  for (auto res : colResistance) {
    double T_col = GetColumnLatency(res);
    double P_col = GetColumnPower(res);
    readDynamicEnergy += T_col * P_col;
  }
  readDynamicEnergy *= numRead;
}

} // namespace CoMN
