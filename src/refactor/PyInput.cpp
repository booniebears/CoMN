/**
 * @file PyInput.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <sstream>

#include "PyInput.h"

namespace Refactor {

PyInput::PyInput() {
  ifstream inputInfo;
  inputInfo.open("../../data_transmiss/PyInput.txt");
  string line;
  if (inputInfo.is_open()) {
    while (getline(inputInfo, line)) {
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
      } else if (key == "input_sparsity") {
        input_sparsity = stod(value);
      }
    }
    // for (auto i : shape) {
    //   cout << i << " ";
    // }
    // cout << endl;
    // cout << "input_sparsity = " << input_sparsity << endl;
  } else {
    throw runtime_error("Cannot Open PyInput.txt!!");
  }
}

} // namespace Refactor
