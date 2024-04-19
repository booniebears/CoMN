/**
 * @file PyWeight.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "PyWeight.h"

namespace Refactor {
PyWeight::PyWeight() {
  ifstream weightInfo;
  weightInfo.open("../../data_transmiss/PyWeight.txt");
  string line;
  if (weightInfo.is_open()) {
    while (getline(weightInfo, line)) {
      istringstream iss(line);
      string key, value;
      getline(iss, key, ':');
      getline(iss, value);

      key.erase(key.find_last_not_of(" ") + 1);
      value.erase(0, key.find_first_not_of(" "));
      if (key == "shape") {
        istringstream ss(value);
        string token;
        while (getline(ss, token, ',')) {
          shape.push_back(stoi(token));
        }
      } else if (key == "weight_sparsity") {
        weight_sparsity = stod(value);
      } else if (key == "type") {
        // "type" is an identifier for Matmul. there are two possible values:
        // "LW" -- weight on the left and "RW" -- weight on the right.
        type = value;
        type.erase(0, type.find_first_not_of(" "));
      }
    }
    // for (auto i : shape) {
    //   cout << i << " ";
    // }
    // cout << endl;
    // cout << "weight_sparsity = " << weight_sparsity << endl;
  } else {
    throw runtime_error("Cannot Open PyWeight.txt!!");
  }
}

} // namespace Refactor
