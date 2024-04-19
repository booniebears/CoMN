/**
 * @file PyParam.h
 * @author booniebears
 * @brief
 * @date 2023-11-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PYPARAM_H_
#define PYPARAM_H_

#include <utility>
using namespace std;
namespace Refactor {

class PyParam {
public:
  PyParam();
  virtual ~PyParam() {}

  int layer;                  // The number of layer in a network
  pair<int, int> stride;      // stride in width and height
  pair<int, int> kernel_size; // kernel_size in width and height
  pair<int, int> padding;     // padding in width and height
  int w_precision;            // weight precision
  int a_precision;            // activation precision
};

} // namespace Refactor

#endif // !PYPARAM_H_