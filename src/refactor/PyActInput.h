/**
 * @file PyActInput.h
 * @author booniebears
 * @brief For input of activations like relu && maxpool
 * @date 2023-11-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PYACTINPUT_H_
#define PYACTINPUT_H_

#include <vector>

using namespace std;

namespace Refactor {
class PyActInput {
public:
  PyActInput();
  virtual ~PyActInput() {}

  vector<int> shape; // shape.size() == 2 || shape.size() == 4
  int mode;
};

} // namespace Refactor

#endif // !PYACTINPUT_H_