#ifndef DEFINES_H_
#define DEFINES_H_

namespace Refactor {

#define PATH_OPT_PARAM "../../../Parameters/OptParam.json"
#define PATH_MACRO_PARAM "../../../Parameters/MacroParam.json"
#define PATH_SPEC_PARAM "../../../Parameters/SpecParam.json"
#define PATH_TECH_PARAM "../../../Parameters/TechnodeParam.json"
#define PATH_TRAFFIC "../../../Parameters/traffic.txt"
#define PATH_PREPARE "../../../Parameters/prepare.txt"
#define PATH_MAPPING "../../../Parameters/mapping.txt"
#define PATH_MESHCONNECT "../../../Parameters/meshconnect.txt"
#define PATH_MESHLAYER "../../../Parameters/Meshlayer.txt"
#define PATH_DUPLICATION                                                       \
  "../../../Parameters/duplication.txt" // original "pipeline.json"
#define PATH_TILENUM "../../../Parameters/tileNum.txt" // "tiles.json"
#define PATH_LAYER "../../../Parameters/layer.txt"     // "layer.json"
#define PATH_PLACING "../../../Parameters/placing.txt"

#define PATH_MACRO_PERF "../../../Performance/MacroPerf.json"
#define PATH_MESH_PERF "../../../Performance/MeshPerf.json"
#define PATH_SPM_PERF "../../../Performance/SPMPerf.json"
#define PATH_HTREE_PERF "../../../Performance/HtreePerf.json"
#define PATH_RELU_PERF "../../../Performance/ReluPerf.json"
#define PATH_MAXPOOL_PERF "../../../Performance/MaxPoolPerf.json"
#define PATH_SIGMOID_PERF "../../../Performance/SigmoidPerf.json"
#define PATH_PERFORMANCE "../../../Performance/performance.txt"
#define PATH_SFU_PERFORMANCE "../../../Performance/SFU_performance.txt"

// #define PATH_ARRAY "../../../CoMN_Sim/build/arrayInfo.csv"
// #define PATH_RELU "../../../CoMN_Sim/build/reluInfo.csv"

// #define DIR_MAPPING "../../../generate_data/mapping_out/"

#define PATH_CACTI_CFG "../../../cacti-master/cache.cfg"
#define PATH_CACTI_CFG_BAK "../../../cacti-master/cache_bak.cfg"
#define PATH_BUFFER "../../../cacti-master/bufferInfo.txt"

#define PATH_ORION_CFG "../../../ORION3_0/SIM_port_cfg.h"
#define PATH_ORION_NEW "../../../ORION3_0/SIM_port.h"
#define PATH_ORION_OUT "../../../ORION3_0/output.txt"

#define PATH_TEST_PARTITION "../test/test_partition.txt"

#define INF 1e9

// Refer to the paper "COMN"
enum Mapping_method {
  COLUMN_TRANS = 0,
  MATRIX_TRANS = 1,
  ROW_TRANS = 2,
  LW_MATMUL_TRANS = 3,
  RW_MATMUL_TRANS = 4
};

enum Pipeline_method {
  RAW_PIPELINE = 0,
  EXTREME_PIPELINE = 1,
  HYBRID_PIPELINE = 2,
  DEFAULT_PIPELINE = 3
};

enum Activation { RELU = 0, MAXPOOL = 1, SIGMOID = 2 };

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

} // namespace Refactor

#endif // !DEFINES_H_