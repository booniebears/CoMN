#include "PyActInput.h"
/**
 * @file PyActInput.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <sstream>

#include "PyActInput.h"

namespace Refactor {

PyActInput::PyActInput() {
  ifstream inputInfo("../../data_transmiss/PyActInput.txt");
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
      } else if (key == "mode") {
        // mode can be 0,1,2,representing relu,maxpool and sigmoid.
        mode = stoi(value);
      }
    }
    // for (auto i : shape) {
    //   cout << i << " ";
    // }
    // cout << endl;
    // cout << "mode = " << mode << endl;
  } else {
    throw runtime_error("Cannot Open PyActInput.txt!!");
  }
}

} // namespace Refactor
