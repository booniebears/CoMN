#include "../include/BasicUnit.h"

namespace CoMN {
BasicUnit::BasicUnit() {
  height = 0;
  width = 0;
  area = 0;
  usedArea = 0;
  readLatency = 0, writeLatency = 0;
  readDynamicEnergy = 0, writeDynamicEnergy = 0;
  leakage = 0;
}
} // namespace CoMN
