/**
 * @file Mapping.h
 * @author booniebears
 * @brief
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MAPPING_H_
#define MAPPING_H_

#include <string>
#include <vector>

#include "PyActInput.h"
#include "PyInput.h"
#include "PyParam.h"
#include "PyWeight.h"
#include "json.hpp"

using json = nlohmann::json;

using namespace std;

namespace Refactor {
// to pass return values from weight_partition
struct PartitionInfo {
  vector<double> Split_tile;
  vector<double> Split_array;
  int Intra_tile;
  int Inter_tile;
  // When no folding is applied, fold_copies = 1;
  // When fold the weights along the y-axis(fold vertically), fold_copies > 0;
  // When fold the weights along the x-axis(fold horizontally), fold_copies < 0;
  int fold_copies;
  int intraRowDup = 1;
  int intraColDup = 1;
};

// to pass return values from Htree_NoC;
struct HtreeNoCInfo {
  vector<int> Tile_NoC;
  int Htree_level;
};

struct TilePerfInfo {
  double area;
  double latency;
  double energy;
};

struct LayerInfo {
  int prelayer = 0;
  int nextlayer = 0;
  // string type; // Type not needed here
  int volumn = 0;
  double prelayer_tile[2] = {0};
  double nextlayer_tile[2] = {0};
};

struct MappingInfo {
  int layer;
  double tile_rows[2];
  double tile_cols[2];
  double Intra_tile[2];
  double tile_nums[2];
  double duplication[2];
};

class Mapping {
private:
  /* data */
public:
  Mapping(PyParam *_pyParam, PyWeight *_pyWeight, PyInput *_pyInput,
          PyActInput *_pyActInput);
  virtual ~Mapping() {}

  void mapping_modules();
  void analysis();
  void auto_mapping();
  void activation_modules();
  void pipeline_optimized();
  vector<vector<int>> Mesh_NoC();
  void Mesh_operation(string user_name, string tid);
  void test_partition();
  void test_HtreeHops();

private:
  void weight_transform(vector<int> &weight_unfold, int &buffer_demand,
                        int features);
  void weight_partition(PartitionInfo &info, vector<int> weight_unfold,
                        int buffer_demand, int duplication);
  void Htree_NoC(HtreeNoCInfo &info, PartitionInfo part_info);
  pair<int, int> Calculate_Buffer(vector<double> Split_tile, int duplication,
                                  int input_vec_num);
  int Calculate_TileNoC(PartitionInfo info, vector<int> Tile_NoC,
                        int duplication, int input_vec_num);
  void Calculate_TilePerformance(TilePerfInfo &tilePerfInfo,
                                 PartitionInfo partitionInfo, int Htree_level,
                                 int HtreeNoC, json htreePerf, json macroPerf,
                                 json spmPerf, int duplication, int input_vec_num);
  void readMappingInfo(ifstream &f, MappingInfo &mappingInfo,
                       LayerInfo layerInfo);
  void writeInfo(double area, double latency, double energy, string mapOutPath,
                 int total_layer);
  void writeTileInfo(int tile_1, int tile_2, int layer, string type,
                     int residual_layers, ofstream &of);

  int adjust_split_array(double &Split_heightArray, double &Split_widthArray);

  void weight_mixed_split(PartitionInfo &info, vector<int> weight_unfold,
                          int buffer_demand, int duplication);

  PyParam *pyParam;
  PyWeight *pyWeight;
  PyInput *pyInput;
  PyActInput *pyActInput;

  bool mapping_optimized;
  int transform_method;
  // int pipeline_method;
  bool prepare_mode;

  int k1, k2, inChannels, outChannels,
      stride; // get relevant params from pyWeight
  int arraySize, bufferSizeTile;
  vector<int> Tile;     // "Tile" in SpecParam.json
  vector<int> Subarray; // "Subarray" in SpecParam.json
  /**
   * Example : Tile_little = {4,4}, Tile_big = {8,8};
   * Suppose Tile_big = Tile_little * 2; And the size of Tile_little can switch
   * from 2 to 8(2,3,4,5,6,7,8).
   */
  int NVM_states;

  int CIM_num;
  int Buffer_read, Buffer_write;
  int TileNoC_num;

  // Test vars
  int layer = 0;
  int HTreeTestCnt = 0;
};

} // namespace Refactor

#endif // !MAPPING_H_