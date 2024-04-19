/**
 * @file PyWeight.h
 * @author booniebears
 * @brief
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PYWEIGHT_H_
#define PYWEIGHT_H_

#include <vector>

using namespace std;
namespace Refactor {
class PyWeight {
public:
  PyWeight();
  virtual ~PyWeight() {}

  vector<int> shape; // shape.size() == 2 || shape.size() == 4
  double weight_sparsity;
  string type = "";
};

} // namespace Refactor

#endif // !PYWEIGHT_H_