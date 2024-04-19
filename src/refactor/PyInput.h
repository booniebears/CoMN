/**
 * @file PyInput.h
 * @author booniebears
 * @brief
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PYINPUT_H_
#define PYINPUT_H_

#include <vector>

using namespace std;

namespace Refactor {

class PyInput {
public:
  PyInput();
  virtual ~PyInput() {}

  // shape.size() == 2 || shape.size() == 4 for linear/Conv/LW Matmul.
  // For RW Matmul, the shape.size() can be veried.
  vector<int> shape; 
  double input_sparsity;
};

} // namespace Refactor

#endif // !PYINPUT_H_