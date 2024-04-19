#ifndef TECHNOLOGY_H_
#define TECHNOLOGY_H_

#include "../../include/general/Types.h"

using namespace CoMN;

namespace CoMN {
class Technology {
public:
  Technology();
  virtual ~Technology() {}

  /* Functions */
  void PrintProperty();
  void Initialize(int _featureSizeInNano, DeviceRoadmap _deviceRoadmap,
                  TransistorType _transistorType);

  /* Properties */
  bool initialized;            /* Initialization flag */
  int featureSizeInNano;       /*Process feature size, Unit: nm */
  double featureSize;          /* Process feature size, Unit: m */
  double RRAMFeatureSize;      /* Process feature size of RRAM, Unit: m */
  DeviceRoadmap deviceRoadmap; /* HP or LP */
  TransistorType transistorType;
  double vdd;           /* Supply voltage, Unit: V */
  double vth;           /* Threshold voltage, Unit: V */
  double heightFin;     /* Fin height, Unit: m */
  double widthFin;      /* Fin width, Unit: m */
  double PitchFin;      /* Fin pitch, Unit: m */
  double phyGateLength; /* Physical gate length, Unit: m */
  double capIdealGate;  /* Ideal gate capacitance, Unit: F/m */
  double capFringe;     /* Fringe capacitance, Unit: F/m */
  double capJunction;   /* Junction bottom capacitance, Cj0, Unit: F/m^2 */
  double capOverlap;    /* Overlap capacitance, Cover in MASTAR, Unit: F/m */
  double capSidewall;   /* Junction sidewall capacitance, Cjsw, Unit: F/m */
  double capDrainToChannel; /* Junction drain to channel capacitance, Cjswg,
                               Unit: F/m */
  double buildInPotential;  /* Bottom junction built-in potential(PB in BSIM4
                               model), Unit: V */
  double pnSizeRatio;       /* PMOS to NMOS size ratio */
  double effectiveResistanceMultiplier; /* Extra resistance due to vdsat */
  double currentOnNmos[101];            /* NMOS saturation current, Unit: A/m */
  double currentOnPmos[101];            /* PMOS saturation current, Unit: A/m */
  double
      currentOffNmos[101]; /* NMOS off current (from 300K to 400K), Unit: A/m */
  double
      currentOffPmos[101]; /* PMOS off current (from 300K to 400K), Unit: A/m */
  double current_gmNmos;   /* NMOS current at 0.7*vdd for gm calculation, Unit:
                              A/m/V*/
  double current_gmPmos;   /* PMOS current at 0.7*vdd for gm calculation, Unit:
                              A/m/V*/

  double capPolywire; /* Poly wire capacitance, Unit: F/m */
};

} // namespace CoMN

#endif