/**
 * @file PyParam.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "PyParam.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace Refactor {

PyParam::PyParam() {
  ifstream paramInfo;
  // relative to build dir
  paramInfo.open("../../data_transmiss/PyParam.txt");
  string line;
  if (paramInfo.is_open()) {
    while (getline(paramInfo, line)) {
      istringstream iss(line);
      string key, value;
      getline(iss, key, ':');
      getline(iss, value);
      // cout << "key = " << key << endl;
      // cout << "value = " << value << endl;
      // remove spaces in key(trailing) and value(leading)
      key.erase(key.find_last_not_of(" ") + 1);
      value.erase(0, key.find_first_not_of(" "));
      if (key == "layer") {
        layer = stoi(value);
      } else if (key == "stride") {
        istringstream ss(value);
        char comma;
        ss >> stride.first >> comma >> stride.second;
      } else if (key == "kernel_size") {
        istringstream ss(value);
        char comma;
        ss >> kernel_size.first >> comma >> kernel_size.second;
      } else if (key == "padding") {
        istringstream ss(value);
        char comma;
        ss >> padding.first >> comma >> padding.second;
      } else if (key == "w_precision") {
        w_precision = stoi(value);
      } else if (key == "a_precision") {
        a_precision = stoi(value);
      }
    }
    // cout << "layer = " << layer << endl;
    // cout << "stride = " << stride.first << "," << stride.second << endl;
    // cout << "kernel_size = " << kernel_size.first << "," <<
    // kernel_size.second << endl; cout << "w_precision = " << w_precision <<
    // endl; cout << "a_precision = " << a_precision << endl;
  } else {
    throw runtime_error("Cannot Open Pyparam.txt!!");
  }
}

} // namespace Refactor
