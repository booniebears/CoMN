#ifndef TYPES_H_
#define TYPES_H_

namespace CoMN {
namespace Type { // To prevent name collision
enum MemCellType {
  SRAM = 1,
  RRAM = 2,
  FeFET = 3,
};
}
enum CellAccessType {
  CMOS_access = 1,
  BJT_access = 2,
  diode_access = 3,
  none_access = 4
};

enum DeviceRoadmap {
  HP = 1,  /* High performance */
  LSTP = 2 /* Low standby power */
};

enum TransistorType {
  Conventional = 1, /* Conventional CMOS */
  FET_2D = 2,       /* 2D FET */
  TFET = 3
};

enum AreaModify {
  NONE = 1,    /* No action, just use the original area calculation */
  MAGIC = 2,   /* Use magic folding based on the original area */
  OVERRIDE = 3 /* directly modify the height and width and calculate new area */
};

enum DecoderMode {
  REGULAR_ROW = 1, /* Regular row mode */
  REGULAR_COL = 2  /* Regular column mode */
};

enum RowColMode {
  ROW_MODE = 1, // Connect to rows
  COL_MODE = 2  // Connect to columns
};

enum ReadCircuitMode {
  CMOS = 1,       /* Normal read circuit */
  OSCILLATION = 2 /* NbO2 */
};

enum SpikingMode {
  NONSPIKING = 1, /* Binary format */
  SPIKING = 2
};

enum BusMode {
  HORIZONTAL = 1, /* horizontal bus */
  VERTICAL = 2,   /* vertical bus */
};
} // namespace CoMN

#endif