#ifndef BASICUNIT_H_
#define BASICUNIT_H_

using namespace std;
namespace CoMN {
class BasicUnit {
public:
  BasicUnit();
  virtual ~BasicUnit() {}

  /* Functions */
  // virtual void PrintProperty(const char *str);
  // virtual void SaveOutput(const char *str);
  // virtual void MagicLayout();
  // virtual void OverrideLayout();

  /* Properties */
  double height;                                /* Unit: m */
  double width;                                 /* Unit: m */
  double area;                                  /* Unit: m^2 */
  double usedArea;                              /* Unit: m^2 */
  double readLatency, writeLatency;             /* Unit: s */
  double readDynamicEnergy, writeDynamicEnergy; /* Unit: J */
  double leakage;                               /* Unit: W */
};
} // namespace CoMN

#endif