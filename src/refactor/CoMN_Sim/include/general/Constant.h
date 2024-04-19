#ifndef CONSTANT_H_
#define CONSTANT_H_

namespace CoMN {
#define INV 0
#define NOR 1
#define NAND 2

#define NMOS 0
#define PMOS 1

#define MAX_NMOS_SIZE 100
#define MIN_NMOS_SIZE 2 

#define MAX_TRANSISTOR_HEIGHT 28
#define MAX_TRANSISTOR_HEIGHT_FINFET 34

#define MIN_GAP_BET_P_AND_N_DIFFS 3.5   // 2
#define MIN_GAP_BET_SAME_TYPE_DIFFS 1.6 // 1.5
#define MIN_GAP_BET_GATE_POLY 2.8       // 1.5
#define MIN_GAP_BET_GATE_POLY_FINFET 3.9
#define MIN_GAP_BET_CONTACT_POLY 0.7 // 0.75
#define CONTACT_SIZE 1.3             // 1
#define MIN_WIDTH_POWER_RAIL 3.4     // 2
#define MIN_POLY_EXT_DIFF 1.0 // Minimum poly extension beyond diffusion region
#define MIN_GAP_BET_FIELD_POLY                                                 \
  1.6 // Field poly means the poly above the field oxide (outside the active
      // region)
#define POLY_WIDTH 1.0
#define POLY_WIDTH_FINFET 1.4
#define M2_PITCH 3.2
#define M3_PITCH 2.8

#define AVG_RATIO_LEAK_2INPUT_NAND 0.48
#define AVG_RATIO_LEAK_3INPUT_NAND 0.31
#define AVG_RATIO_LEAK_2INPUT_NOR 0.95
#define AVG_RATIO_LEAK_3INPUT_NOR 0.62

#define W_SENSE_P 7.5
#define W_SENSE_N 3.75
#define W_SENSE_ISO 12.5
#define W_SENSE_EN 5.0
#define W_SENSE_MUX 9.0

#define IR_DROP_TOLERANCE 0.25
#define LINEAR_REGION_RATIO 0.20

#define HEIGHT_WIDTH_RATIO_LIMIT 5

#define RATIO_READ_THRESHOLD_VS_VOLTAGE 0.2

#define INF_RAMP 1e20 // infinity ramp input (rising edge of voltage)
#define MAXPOOL_WINDOW 2*2
} // namespace CoMN

#endif