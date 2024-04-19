#ifndef MEMCELL_H_
#define MEMCELL_H_

#include "../../include/general/Types.h"

namespace CoMN {
class MemCell {
public:
  /* Properties */
  Type::MemCellType memCellType; /* Memory cell type (like MRAM, PCRAM, etc.) */
  int processNode;    /* Cell original process technology node, Unit: nm*/
  double area;        /* Cell area, Unit: F^2 */
  double aspectRatio; /* Cell aspect ratio, H/W */
  double widthInFeatureSize;  /* Cell width, Unit: F */
  double heightInFeatureSize; /* Cell height, Unit: F */
  double resistanceOn;        /* Turn-on resistance */
  double resistanceOff;       /* Turn-off resistance */
  double minSenseVoltage;     /* Minimum sense voltage */

  CellAccessType accessType; /* Cell access type: CMOS, BJT, or diode */
  double featureSize;
  double accessVoltage;
  double readVoltage;
  double writeVoltage;
  double readPulseWidth;
  double writePulseWidth;
  double resistanceAccess;
  bool nonlinearIV; /* Consider I-V nonlinearity or not (Currently this option
                       is for cross-point array. It is hard to have this option
                       in pseudo-crossbar since it has an access transistor and
                       the transistor's resistance can be comparable to RRAM's
                       resistance after considering the nonlinearity. In this
                       case, we have to iteratively find both the resistance and
                       Vw across RRAM.) */
  double nonlinearity; /* Current at write voltage / current at 1/2 write
                          voltage */
  double resistanceAvg;
  double resCellAccess;
  double resMemCellOn;  // At on-chip Vr (different than the Vr in the reported
                        // measurement data)
  double resMemCellOff; // At on-chip Vr (different than the Vr in the reported
                        // measurement data)
  double resMemCellAvg; // At on-chip Vr (different than the Vr in the reported
                        // measurement data)
  double resMemCellOnAtHalfVw;
  double resMemCellOffAtHalfVw;
  double resMemCellAvgAtHalfVw;
  double resMemCellOnAtVw;
  double resMemCellOffAtVw;
  double resMemCellAvgAtVw;
  double capSRAMCell;
  int multipleCells; /* Use multiple cells as one weight element to reduce the
                        variation (only layout now) */
  int maxNumLevelLTP, maxNumLevelLTD;

  /* Optional properties */
  double
      widthAccessCMOS; /* The gate width of CMOS access transistor, Unit: F */
  double widthSRAMCellNMOS; /* The gate width of NMOS in SRAM cells, Unit: F */
  double widthSRAMCellPMOS; /* The gate width of PMOS in SRAM cells, Unit: F */
};

} // namespace CoMN

#endif