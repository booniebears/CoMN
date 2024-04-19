/**
 * @file Mapping.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <math.h>
#include <set>

#include "Mapping.h"
#include "Mesh_Placing.h"
#include "defines.h"

namespace Refactor {

bool testMode = false;

Mapping::Mapping(PyParam *_pyParam, PyWeight *_pyWeight, PyInput *_pyInput,
                 PyActInput *_pyActInput)
    : pyParam(_pyParam), pyWeight(_pyWeight), pyInput(_pyInput),
      pyActInput(_pyActInput) {
  ifstream f(PATH_OPT_PARAM);
  json optParam = json::parse(f);
  ifstream f_macro(PATH_MACRO_PARAM);
  json macroParam = json::parse(f_macro);
  mapping_optimized = optParam["mapping_optimized"];
  transform_method = optParam["mapping_method"];
  if (pyWeight->type == "LW") { // switch type.
    transform_method = LW_MATMUL_TRANS;
  } else if (pyWeight->type == "RW") {
    transform_method = RW_MATMUL_TRANS;
  }
  // pipeline_method = optParam["pipeline_method"];
  prepare_mode = optParam["prepare_mode"];
  stride = pyParam->stride.first;
  // k = (pyWeight->shape.size() == 4) ? pyParam->kernel_size.first : 1;
  k1 = (pyWeight->shape.size() == 4) ? pyParam->kernel_size.first : 1;
  k2 = (pyWeight->shape.size() == 4) ? pyParam->kernel_size.second : 1;
  if (transform_method == LW_MATMUL_TRANS) {
    inChannels = pyWeight->shape[1];
    outChannels = pyWeight->shape[0];
  } else if (transform_method == RW_MATMUL_TRANS) {
    inChannels = pyWeight->shape[0];
    outChannels = pyWeight->shape[1];
  } else {
    inChannels = pyWeight->shape[1];
    outChannels = pyWeight->shape[0];
  }
  NVM_states = macroParam["NVM_states"];
}

void Mapping::mapping_modules() {
  if (prepare_mode) {
    analysis();
    // cout << "Analysis Done!!!" << endl;
  } else {
    auto_mapping();
    // cout << "auto_mapping Done!!!" << endl;
  }
}

/**
 * @brief Compute array_size(int or double?) and compute_cycle, and write into
 * Parameters/prepare.txt in "a+" format.
 *
 */
void Mapping::analysis() {
  int compute_cycle = 1, weight_features = 1;
  for (auto i : pyWeight->shape) {
    weight_features *= i;
  }
  double array_size = weight_features * 2 * pyParam->w_precision / NVM_states;
  if (transform_method == LW_MATMUL_TRANS) {
    // Matrix mul, Weight on the left;
    for (int i = 1; i < pyInput->shape.size(); i++) {
      compute_cycle *= pyInput->shape[i];
    }
  } else if (transform_method == RW_MATMUL_TRANS) {
    // Matrix mul, Weight on the right;
    for (int i = 0; i < pyInput->shape.size() - 1; i++) {
      compute_cycle *= pyInput->shape[i];
    }
  } else if (pyWeight->shape.size() == 4) {
    // Conv layers
    compute_cycle = ((pyInput->shape[2] + 2 * pyParam->padding.first -
                      pyParam->kernel_size.first) / pyParam->stride.first + 1) *
                    ((pyInput->shape[3] + 2 * pyParam->padding.second -
                      pyParam->kernel_size.second) / pyParam->stride.second + 1);
  } else if (pyWeight->shape.size() == 2) {
    // Linear layers
    compute_cycle = 1;
  } else {
    throw runtime_error("In analysis(), unidentified layer type found!!!");
  }
  // Attach current layer's info to the file.
  ofstream prepare(PATH_PREPARE, ios::app);
  prepare << setprecision(16);
  prepare << "layer: " << pyParam->layer << " array_size: " << array_size
          << " compute_cycle: " << compute_cycle << endl;
  prepare.close();
}

/**
 * @brief Map the designated weight onto Tiles, and calculate performance
 * information including area/latency/energy.
 * Following the order of : weight_transform -> weight_partition -> Htree_NoC ->
 * (CIM_num) -> Calculate_Buffer
 *
 */
void Mapping::auto_mapping() {
  // Read json files and other data files and prepare for mapping
  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  ifstream f_macro(PATH_MACRO_PERF);
  json macroPerf = json::parse(f_macro);
  ifstream f_spm(PATH_SPM_PERF);
  json spmPerf = json::parse(f_spm);
  ifstream f_htree(PATH_HTREE_PERF);
  json htreePerf = json::parse(f_htree);
  vector<int> duplication; // duplication for each layer
  int val;
  ifstream f(PATH_DUPLICATION);
  while (f >> val) {
    duplication.push_back(val);
  }
  int curDup = duplication[pyParam->layer - 1];
  arraySize = specParam["Subarray"][0]; // side length of a subarray.
  bufferSizeTile = specParam["buffersizeTile"];
  bufferSizeTile *= 1024 * 8; // KB -> b
  Tile.push_back(specParam["Tile"][0]);
  Tile.push_back(specParam["Tile"][1]);
  Subarray.push_back(specParam["Subarray"][0]);
  Subarray.push_back(specParam["Subarray"][1]);
  // cout << "Value Set Done!!" << endl;
  // Adjust the Mapping method according to Weight Size
  if (mapping_optimized) {
    if (k1 * k2 * inChannels < arraySize) {
      transform_method = COLUMN_TRANS; // [k1*k2*inChannels,outChannels]
    } else if (inChannels > 2 * outChannels) {
      transform_method = ROW_TRANS; // [inChannels,k1*k2*outChannels]
    } else {
      transform_method = MATRIX_TRANS; // [k1*inChannels,k2*outChannels]
    }
  }

  vector<int> weight_unfold;         // 2 nums, in column and row dimension
  int buffer_demand;                 // Buffer for NN inputs of each layer
  if (pyWeight->shape.size() == 2) { // Matmul && Linear
    weight_unfold.push_back(inChannels);
    weight_unfold.push_back(outChannels);
    buffer_demand = inChannels * pyParam->a_precision;
  } else {
    weight_transform(weight_unfold, buffer_demand, pyInput->shape[3]);
  }
  // cout << "weight_transform Done!!" << endl;
  PartitionInfo partitionInfo;
  weight_partition(partitionInfo, weight_unfold, buffer_demand, curDup);
  // cout << "weight_partition Done!!" << endl;
  HtreeNoCInfo htreeNoCInfo;
  Htree_NoC(htreeNoCInfo, partitionInfo);
  // cout << "Htree_NoC Done!!" << endl;
  // modify pyInput shape in terms of linear layer or matmul layer
  if (pyWeight->shape.size() == 2) {
    pyInput->shape.resize(4, 1); // (a,b) -> (a,b,1,1)
  }

  int input_vec_num = 1; // num of input vecs to Subarrays for Matmul
  if (transform_method == LW_MATMUL_TRANS) {
    for (int i = 1; i < pyInput->shape.size(); i++) {
      input_vec_num *= pyInput->shape[i];
    }
  } else if (transform_method == RW_MATMUL_TRANS) {
    for (int i = 0; i < pyInput->shape.size() - 1; i++) {
      input_vec_num *= pyInput->shape[i];
    }
  } else {
    input_vec_num = pyInput->shape[2] * pyInput->shape[3] / (stride * stride);
  }

  // Num of CIM arrays used when processing a layer. Note that arrays are not
  // fully utilized, so the values of Split_array[] are important.
  CIM_num = partitionInfo.Split_array[0] * partitionInfo.Split_array[1] *
            partitionInfo.Inter_tile * partitionInfo.Intra_tile *
            input_vec_num / curDup * pyParam->a_precision;
  // if (pyWeight->type == "LW") {
  //   int input_vec_num = 1; // num of input vecs to Subarrays
  //   for (int i = 1; i < pyInput->shape.size(); i++) {
  //     input_vec_num *= pyInput->shape[i];
  //   }
  //   CIM_num = partitionInfo.Split_array[0] * partitionInfo.Split_array[1] *
  //             partitionInfo.Inter_tile * partitionInfo.Intra_tile *
  //             input_vec_num / curDup * pyParam->a_precision;
  // } else if (pyWeight->type == "RW") {
  //   int input_vec_num = 1; // num of input vecs to Subarrays
  //   for (int i = 0; i < pyInput->shape.size() - 1; i++) {
  //     input_vec_num *= pyInput->shape[i];
  //   }
  //   CIM_num = partitionInfo.Split_array[0] * partitionInfo.Split_array[1] *
  //             partitionInfo.Inter_tile * partitionInfo.Intra_tile *
  //             input_vec_num / curDup * pyParam->a_precision;
  // } else {
  //   CIM_num = partitionInfo.Split_array[0] * partitionInfo.Split_array[1] *
  //             partitionInfo.Inter_tile * partitionInfo.Intra_tile *
  //             pyInput->shape[2] * pyInput->shape[3] / (stride * stride) /
  //             curDup * pyParam->a_precision;
  // }

  auto Buffer_result = Calculate_Buffer(partitionInfo.Split_tile, curDup, input_vec_num);
  Buffer_read = Buffer_result.first, Buffer_write = Buffer_result.second;
  TileNoC_num = Calculate_TileNoC(partitionInfo, htreeNoCInfo.Tile_NoC, curDup,
                                  input_vec_num);
  TilePerfInfo tilePerfInfo;
  int HtreeNoC = specParam["HtreeNoC_flitband"];
  HtreeNoC *= 8;
  Calculate_TilePerformance(tilePerfInfo, partitionInfo,
                            htreeNoCInfo.Htree_level, HtreeNoC, htreePerf,
                            macroPerf, spmPerf, curDup, input_vec_num);
  // cout << "Calculate_TilePerformance Done!!" << endl;
  // Mesh_NoC: call once is enough.
  // vector<vector<int>> traffic = Mesh_NoC();
  // cout << "Mesh_NoC Done!!" << endl;
  ofstream of_perf(PATH_PERFORMANCE, ios::app);
  of_perf << setprecision(16); // set precision
  of_perf << "layer: " << pyParam->layer << " energy: " << tilePerfInfo.energy
          << " latency: " << tilePerfInfo.latency
          << " area: " << tilePerfInfo.area << endl;
  of_perf.close();
  // ofstream of_traffic(PATH_TRAFFIC, ios::app);
  // for (auto vec : traffic) {
  //   for (auto id : vec) {
  //     of_traffic << id << " ";
  //   }
  //   of_traffic << endl;
  // }
  // of_traffic.close();
}

void Mapping::activation_modules() {
  // When prepare_mode == true, conduct mapping_modules() -- analysis();
  // When prepare_mode == false, conduct activation_modules() and
  // mapping_modules() -- auto_mapping(). conducted after auto_mapping().
  if (!prepare_mode) {
    // SFU = special function unit
    ifstream f_spec(PATH_SPEC_PARAM);
    json specParam = json::parse(f_spec);
    ifstream f_macro(PATH_MACRO_PARAM);
    json macroParam = json::parse(f_macro);
    int act_bits = macroParam["ADC_resolution"];
    int SFU_num = specParam["SFU_num"];
    ifstream f_reluPerf(PATH_RELU_PERF);
    json reluPerf = json::parse(f_reluPerf);
    ifstream f_maxPoolPerf(PATH_MAXPOOL_PERF);
    json maxPoolPerf = json::parse(f_maxPoolPerf);
    ifstream f_sigmoidPerf(PATH_SIGMOID_PERF);
    json sigmoidPerf = json::parse(f_sigmoidPerf);
    ifstream f(PATH_MAPPING);
    string line;
    vector<double> Split_tile;
    while (getline(f, line)) {
      istringstream iss(line);
      vector<string> tokens;
      string token;
      while (iss >> token) {
        tokens.push_back(token);
      }
      if (pyParam->layer == stoi(tokens[1])) {
        Split_tile.push_back(stod(tokens[3]));
        Split_tile.push_back(stod(tokens[5]));
      }
    }
    int volumn = 1;
    double area, latency, energy;
    for (auto i : pyActInput->shape) {
      volumn *= i;
    }
    if (pyActInput->mode == RELU) {
      double reluArea = reluPerf["area"];
      double reluLatency = reluPerf["latency"];
      double reluEnergy = reluPerf["energy"];
      area = reluArea * SFU_num * Split_tile[1] * act_bits;
      latency = volumn * reluLatency / SFU_num / Split_tile[1];
      energy = volumn * reluEnergy * act_bits;
    } else if (pyActInput->mode == MAXPOOL) {
      double maxPoolArea = maxPoolPerf["area"];
      double maxPoolLatency = maxPoolPerf["latency"];
      double maxPoolEnergy = maxPoolPerf["energy"];
      area = maxPoolArea * SFU_num * Split_tile[1] * act_bits;
      latency = volumn * maxPoolLatency / SFU_num / Split_tile[1];
      energy = volumn * maxPoolEnergy * act_bits;
    } else {
      // Sigmoid!
      double sigmoidArea = sigmoidPerf["area"];
      double sigmoidLatency = sigmoidPerf["latency"];
      double sigmoidEnergy = sigmoidPerf["energy"];
      area = sigmoidArea * SFU_num * Split_tile[1] * act_bits;
      latency = volumn * sigmoidLatency / SFU_num / Split_tile[1];
      energy = volumn * sigmoidEnergy * act_bits;
    }
    string SFU_mode = pyActInput->mode == RELU      ? "RELU"
                      : pyActInput->mode == MAXPOOL ? "MAXPOOL"
                                                    : "SIGMOID";
    ofstream of(PATH_SFU_PERFORMANCE, ios::app);
    of << setprecision(16);
    of << "layer: " << pyParam->layer << " SFU_energy: " << energy
       << " SFU_latency: " << latency << " SFU_area: " << area
       << " SFU_mode: " << SFU_mode << endl;
  }
}

/**
 * @brief Use info from prepare.txt to calculate duplication of each layer.
 * array_size and compute_cycle are recorded in prepare.txt.
 */
void Mapping::pipeline_optimized() {
  ifstream f(PATH_OPT_PARAM);
  json optParam = json::parse(f);
  int pipeline_method;
  if (optParam["latency"] == false && optParam["area"] == false) {
    pipeline_method = DEFAULT_PIPELINE;
  } else if (optParam["latency"] == true && optParam["area"] == false) {
    pipeline_method = EXTREME_PIPELINE;
  } else if (optParam["latency"] == false && optParam["area"] == true) {
    pipeline_method = RAW_PIPELINE;
  } else {
    pipeline_method = HYBRID_PIPELINE;
  }
  f.close();
  optParam["pipeline_method"] = pipeline_method;
  ofstream of(PATH_OPT_PARAM);
  of << std::setw(2) << optParam << std::endl;
  of.close();
  cout << "Saving pipeline_method!!" << endl;
  // Reading layer.txt
  ifstream f_layer(PATH_LAYER);
  // Attention: conv_layer here means "not Linear layer" after introducing
  // matmul operator.
  int conv_layer, total_layer;
  f_layer >> conv_layer >> total_layer;

  ifstream f_prepare(PATH_PREPARE);
  string line;
  vector<int> duplication(total_layer, 1), compute_cycle(conv_layer);
  vector<double> array_size(conv_layer);
  vector<string> tokens;

  cout << "Start dealing with pipeline!!" << endl;
  if (pipeline_method == DEFAULT_PIPELINE || pipeline_method == RAW_PIPELINE) {
    while (getline(f_prepare, line)) {
      istringstream iss(line);
      string token;
      tokens.clear();
      while (iss >> token) {
        tokens.push_back(token);
      }
      int cur_layer = stoi(tokens[1]);
      if (cur_layer <= conv_layer) {
        duplication[cur_layer - 1] = 1;
      }
    }
  }

  else if (pipeline_method == EXTREME_PIPELINE) {
    int min_compute_cycles = INF;
    while (getline(f_prepare, line)) {
      istringstream iss(line);
      string token;
      tokens.clear();
      while (iss >> token) {
        tokens.push_back(token);
      }
      int cur_layer = stoi(tokens[1]);
      if (cur_layer <= conv_layer) {
        compute_cycle[cur_layer - 1] = stoi(tokens[5]);
        min_compute_cycles =
            min(min_compute_cycles, compute_cycle[cur_layer - 1]);
      }
    }
    for (int l = 0; l < conv_layer; l++) {
      duplication[l] = compute_cycle[l] / min_compute_cycles;
    }
  }

  else if (pipeline_method == HYBRID_PIPELINE) {
    while (getline(f_prepare, line)) {
      istringstream iss(line);
      string token;
      tokens.clear();
      while (iss >> token) {
        tokens.push_back(token);
      }
      int cur_layer = stoi(tokens[1]);
      if (cur_layer <= conv_layer) {
        array_size[cur_layer - 1] = stod(tokens[3]);
        compute_cycle[cur_layer - 1] = stoi(tokens[5]);
      }
    }
    vector<int> tmp_duplication(total_layer, 1),
        tmp_compute_cycle = compute_cycle;
    vector<double> tmp_array_size = array_size;
    // Test the most suitable layer to calculate duplication nums
    double lowest_score = 1e20;
    for (int i = conv_layer - 1; i >= 0; i--) {
      array_size = tmp_array_size;
      compute_cycle = tmp_compute_cycle;
      fill(tmp_duplication.begin(), tmp_duplication.end(), 1);
      for (int j = 0; j < conv_layer; j++) {
        if (compute_cycle[j] >= compute_cycle[i]) {
          tmp_duplication[j] = compute_cycle[j] / compute_cycle[i];
          compute_cycle[j] = compute_cycle[i];
          array_size[j] = array_size[j] * tmp_duplication[j];
        }
      }
      double score = accumulate(array_size.begin(), array_size.end(), 0.0) *
                     compute_cycle[i];
      if (score < lowest_score) {
        lowest_score = score;
        duplication = tmp_duplication;
        // cout << "Current i = " << i << endl;
      }
    }
  }
  ofstream of_dup(PATH_DUPLICATION);
  for (auto dup : duplication) {
    of_dup << dup << " ";
  }
  of_dup.close();
}

/**
 * @brief Get the total_energy,total_latency,total_area
 *
 */
void Mapping::Mesh_operation(string user_name, string tid) {
  // cout << "Into Mesh_operation!!!" << endl;
  ifstream f(PATH_OPT_PARAM);
  json optParam = json::parse(f);
  int pipeline_method = optParam["pipeline_method"];

  ifstream f_traffic(PATH_TRAFFIC);
  // vector<vector<int>> traffic;
  map<pair<int, int>, int> traffic_mp; // The data transfer between x and y.
  int src, dst, packet;
  // while (f_traffic >> src >> dst >> packet) {
  //   traffic.push_back({src, dst, packet});
  // }
  while (f_traffic >> src >> dst >> packet) {
    traffic_mp[{src, dst}] += packet;
  }

  ifstream f_mesh(PATH_MESH_PERF);
  json meshPerf = json::parse(f_mesh);
  double meshArea = meshPerf["area"], meshLatency = meshPerf["latency"],
         meshEnergy = meshPerf["energy"];

  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  int MeshNoC = specParam["MeshNoC_flitband"];
  MeshNoC *= 8;

  ifstream f_tile(PATH_TILENUM);
  int tileNum;
  f_tile >> tileNum;
  // tileNum++;
  int conv_layer, total_layer;
  ifstream f_layer(PATH_LAYER);
  f_layer >> conv_layer >> total_layer; // conv_layer not used here.
  Mesh_Placing *placing =
      new Mesh_Placing(MeshNoC, MeshNoC, tileNum, meshLatency);

  MeshInfo meshInfo;
  if (tid == "spec") {
    meshInfo = placing->Mesh_mapping_random_pipeline(traffic_mp);
  } else {
    meshInfo = placing->Mesh_mapping_energy_pipeline(traffic_mp);
  }

  double MeshNoC_energy = meshInfo.link_length * meshEnergy;
  double MeshNoC_latency = meshInfo.Mesh_latency;
  double MeshNoC_area = meshArea * tileNum;
  if (pipeline_method == DEFAULT_PIPELINE) {
    MeshNoC_latency = MeshNoC_latency * total_layer;
  }

  // performance.txt does not include SFU!!
  ifstream f_perf(PATH_PERFORMANCE);
  double totalEnergy = MeshNoC_energy;
  double totalLatency = 0;
  double totalArea = MeshNoC_area;
  string line;
  // cout << "First step in Calculating perf!!!" << endl;
  while (getline(f_perf, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    // Filter the "total energy" line
    if (tokens.size() != 8 || tokens[0] != "layer:")
      continue;
    double tmpEnergy = stod(tokens[3]);
    double tmpLatency = stod(tokens[5]);
    double tmpArea = stod(tokens[7]);
    if (pipeline_method == DEFAULT_PIPELINE) {
      totalLatency += tmpLatency;
    } else {
      totalLatency = max(totalLatency, tmpLatency);
    }
    totalArea += tmpArea;
    totalEnergy += tmpEnergy;
  }
  // cout << "totalArea = " << totalArea << endl;
  // cout << "totalLatency = " << totalLatency << endl;
  // cout << "totalEnergy = " << totalEnergy << endl;
  f_perf.close();

  ifstream f_SFUPerf(PATH_SFU_PERFORMANCE);
  double SFULatency = 0;
  while (getline(f_SFUPerf, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    double SFUEnergy = stod(tokens[3]);
    double tmpLatency = stod(tokens[5]);
    double SFUArea = stod(tokens[7]);
    if (pipeline_method == DEFAULT_PIPELINE) {
      SFULatency += tmpLatency;
    } else {
      SFULatency = max(SFULatency, tmpLatency);
    }
    totalArea += SFUArea;
    totalEnergy += SFUEnergy;
  }
  totalLatency += SFULatency + MeshNoC_latency;
  // cout << "totalArea = " << totalArea << endl;
  // cout << "totalLatency = " << totalLatency << endl;
  // cout << "totalEnergy = " << totalEnergy << endl;
  ofstream of(PATH_PERFORMANCE, ios::app);
  of << setprecision(16);
  of << endl;
  of << "total energy: " << totalEnergy << " total latency: " << totalLatency
     << " total area: " << totalArea << endl;
  of.close();
  free(placing);
  if (optParam["specification_optimized"] == false &&
      optParam["circuit_optimized"] == false) {
    string mapOutPath = "../../../generate_data/" + user_name +
                        "/mapping_out/mapping_out" + tid + ".txt";
    writeInfo(totalArea, totalLatency, totalEnergy, mapOutPath, total_layer);
  }
}

/**
 * @brief Calculate the width and height of weight unfolded in Memcells,
 * and store relevant info in Parameters/mapping.txt.
 * @param features : input.shape[3], the width of input feature
 *
 */
void Mapping::weight_transform(vector<int> &weight_unfold,
                               int &buffer_demand, int features) {
  if (transform_method == COLUMN_TRANS) {
    // buffer_demand for COLUMN_TRANS = ((Nx × (Ky − 1)) + Kx) × Nif.
    // Nx = number of rows in the input feature map;
    // Ky and Kx = the number of columns and rows in the kernel;
    // Nif = number of input feature maps involved in the convolution step
    weight_unfold.push_back(k1 * k2 * inChannels);
    // double weights are required to represent signed values.
    weight_unfold.push_back(outChannels * pyParam->w_precision / NVM_states *
                            2);

    // Kx = Ky = (k - 1) / stride + 1;
    buffer_demand = (features * (k1 - 1) / stride + (k1 - 1) / stride + 1) *
                    inChannels * pyParam->a_precision;
  } else if (transform_method == MATRIX_TRANS) {
    weight_unfold.push_back(k1 * inChannels);
    weight_unfold.push_back(k2 * outChannels * pyParam->w_precision /
                            NVM_states * 2);
    buffer_demand =
        (features * (k1 - 1) / stride + 1) * inChannels * pyParam->a_precision;
  } else if (transform_method == ROW_TRANS) {
    weight_unfold.push_back(inChannels);
    weight_unfold.push_back(k1 * k2 * outChannels * pyParam->w_precision /
                            NVM_states * 2);
    buffer_demand = (features * (k1 - 1) / stride + (k1 - 1) / stride + 1) *
                    outChannels * pyParam->a_precision;
  }
}

/**
 * @brief Partition the weights into tiles and subarrays.
 *
 */
void Mapping::weight_partition(PartitionInfo &info, vector<int> weight_unfold,
                               int buffer_demand, int duplication) {
  // We consider folding weights in partition. More specifically, when the
  // width/height of "weight_unfold" is greater than the size of a tile, try to
  // fold it into fewer tiles to increase utilization rate.

  // Caution: After folding, the HTree routing conditions will change, so in
  // HTree_NoC, the cost of routing has to be discussed in different scenarios.

  double Split_heightArray = (double)weight_unfold[0] / Subarray[0];
  double Split_widthArray = (double)weight_unfold[1] / Subarray[1];

  int intraTile, fold_copies;
  fold_copies = adjust_split_array(Split_heightArray, Split_widthArray);

  // Duplication has to be considered. The placement of duplication also
  // matters. When we place duplications in rows or columns, the buffer
  // read/write energy may change a lot.

  // Here, we assume that all duplications are placed in a column across tiles.
  // The partition may optimize after proposing a detailed "target function".

  // Duplication num in Row/Column
  int maxRowDup = MAX(MIN(Tile[0], floor(Tile[0] / Split_heightArray)), 1);
  int maxColDup = MAX(MIN(Tile[1], floor(Tile[1] / Split_widthArray)), 1);
  intraTile = maxRowDup * maxColDup; // Max duplications in a Tile
  // BufferSize also needs to be taken into consideration. Duplication in a tile
  // cannot exceed the buffer size.
  intraTile = MIN(MAX(1, bufferSizeTile / buffer_demand), intraTile);
  intraTile = MIN(intraTile, duplication); // cannot exceed "duplication".
  maxRowDup = ceil(sqrt((double)intraTile));
  while (intraTile % maxRowDup != 0) {
    maxRowDup--;
  }
  info.intraRowDup = maxRowDup;
  info.intraColDup = intraTile / maxRowDup;

  // Duplication num across tiles (apart from duplication in a tile).
  int TileDups = ceil((double)duplication / intraTile);
  // ++layer;
  // if (layer == 2) {
  //   cout << "layer = " << layer << " Split_heightArray = " <<
  //   Split_heightArray
  //        << " Split_widthArray = " << Split_widthArray << endl;
  //   cout << "TileDups = " << TileDups << endl;
  // }

  // Duplications are stacked in rows.
  double Split_heightTile =
      ceil(MAX(Split_heightArray, Tile[0]) * TileDups / Tile[0]);
  double Split_widthTile = ceil(Split_widthArray / Tile[1]);
  Split_heightArray =
      Split_heightArray * TileDups / Split_heightTile; // resize to (0,Tile[0]]
  Split_widthArray /= Split_widthTile;                 // resize to (0,Tile[1]]
  int interTile = Split_heightTile * Split_widthTile;
  // if (layer == 2) {
  //   cout << "Split_heightTile = " << Split_heightTile
  //        << " Split_widthTile = " << Split_widthTile << endl;
  // }
  double demand_tiles = ceil((double)buffer_demand / bufferSizeTile);
  if (demand_tiles > interTile) {
    // The tiles already split do not have enough buffer:
    Split_heightTile = ceil(sqrt(demand_tiles));
    Split_widthTile = ceil(demand_tiles / Split_heightTile);
    // TODO: some problems with Split array???
    Split_heightArray =
        (double)weight_unfold[0] / Subarray[0] / Split_heightTile;
    Split_widthArray = (double)weight_unfold[1] / Subarray[1] / Split_widthTile;
    fold_copies = adjust_split_array(Split_heightArray, Split_widthArray);
    interTile = Split_heightTile * Split_widthTile;
  }

  // Split_tile is exactly the split of tiles in width/height considering
  // duplication; Split_array is the split of subarray in width/height without
  // considering duplication in tile!!! (but the duplication between tiles are
  // considered)
  info.Split_array.resize(2);
  info.Split_tile.resize(2);
  info.Split_array[0] = Split_heightArray;
  info.Split_array[1] = Split_widthArray;
  // Duplications are considered in Split_tile.
  info.Split_tile[0] = Split_heightTile;
  info.Split_tile[1] = Split_widthTile;

  // intraTile = pow(2, floor(log2(intraTile))); // resize to 2^n
  // if (intraTile > 1) { // One tile can store all the weights(no duplication)
  //   if (intraTile <= duplication) {
  //     interTile = ceil(duplication / intraTile);
  //   } else {
  //     intraTile = duplication;
  //     interTile = 1;
  //   }
  //   double Split_heightArray = (double)weight_unfold[0] / Subarray[0];
  //   double Split_widthArray = (double)weight_unfold[1] / Subarray[1];
  //   double Split_heightTile = 1, Split_widthTile = 1;
  //   if (Split_heightArray > Tile[0]) {
  //     Split_heightTile = ceil(Split_heightArray / Tile[0]);
  //   } else if (Split_widthArray > Tile[1]) {
  //     Split_widthTile = ceil(Split_widthArray / Tile[1]);
  //   }
  //   // while (Split_heightArray > Tile[0]) {
  //   //   Split_heightArray /= 2;
  //   //   Split_widthArray *= 2;
  //   // }
  //   // while (Split_widthArray > Tile[1]) {
  //   //   Split_widthArray /= 2;
  //   //   Split_heightArray *= 2;
  //   // }

  //   // Split_tile has to be integer; Split_array[0] && Split_array[1]
  //   represent
  //   // the ratio of
  //   info.Split_array.push_back(Split_heightArray);
  //   info.Split_array.push_back(Split_widthArray);
  //   info.Split_tile.push_back(Split_heightTile);
  //   info.Split_tile.push_back(Split_widthTile);
  //   // cout << "weight_unfold = " << weight_unfold[0] << "," <<
  //   weight_unfold[1]
  //   //      << endl;
  // } else { // Multiple Tiles are required to store the weights
  //   int Lsplit, Wsplit;
  //   if (pyWeight->shape.size() == 4) {
  //     // TODO: Some questions about using Lsplit to calculate Split_tile.
  //     // Split_tile should be calculated by dividing Tile[0] * Subarray[0]?
  //     Lsplit = ceil(
  //         sqrt(tileSize * k * k * NVM_states / (2 * pyParam->w_precision)));
  //   } else {
  //     Lsplit = ceil(sqrt(tileSize * NVM_states / (2 *
  //     pyParam->w_precision)));
  //   }
  //   // Lspilt should be divisible by arraySize
  //   Lsplit = MAX(floor(Lsplit / Subarray[0]), 1) * Subarray[0];
  //   Wsplit = floor(ceil(tileSize / Lsplit) / Subarray[1]) * Subarray[1];
  //   info.Split_tile.push_back(ceil((double)weight_unfold[0] / Lsplit));
  //   info.Split_tile.push_back(ceil((double)weight_unfold[1] / Wsplit));
  //   if (info.Split_tile[0] * info.Split_tile[1] <
  //       ceil(buffer_demand / bufferSizeTile)) {
  //     info.Split_tile[0] = ceil(sqrt((double)buffer_demand /
  //     bufferSizeTile)); info.Split_tile[1] = ceil(sqrt((double)buffer_demand
  //     / bufferSizeTile));
  //   }
  //   intraTile = 1;
  //   interTile = info.Split_tile[0] * info.Split_tile[1] * duplication;
  //   info.Split_array.push_back(MIN((double)Lsplit / Subarray[0],
  //                                  (double)weight_unfold[0] / Subarray[0]));
  //   info.Split_array.push_back(MIN((double)Wsplit / Subarray[1],
  //                                  (double)weight_unfold[1] / Subarray[1]));
  // }

  info.Intra_tile = intraTile;
  info.Inter_tile = interTile;
  info.fold_copies = fold_copies;

  if (!testMode) {
    ofstream of(PATH_MAPPING, ios::app); // adding
    of << setprecision(16);
    of << "layer: " << pyParam->layer
       << " Split_tile[0]: " << info.Split_tile[0]
       << " Split_tile[1]: " << info.Split_tile[1]
       << " Split_array[0]: " << info.Split_array[0]
       << " Split_array[1]: " << info.Split_array[1]
       << " intraTile: " << intraTile << " interTile: " << interTile
       << " duplication: " << duplication << endl;
    of.close();
  } else {
    ofstream of(PATH_TEST_PARTITION, ios::app); // adding
    of << setprecision(16);
    of << " Split_tile[0]: " << info.Split_tile[0]
       << " Split_tile[1]: " << info.Split_tile[1]
       << " Split_array[0]: " << info.Split_array[0]
       << " Split_array[1]: " << info.Split_array[1]
       << " intraTile: " << intraTile << " interTile: " << interTile
       << " duplication: " << duplication << " fold_copies: " << fold_copies
       << endl;
    of.close();
  }
}

/**
 * @brief Adjust the value of Split_heightArray/Split_widthArray by folding
 * weight into many halves. Currently, we fold the weight only when one side of
 * weight <= 0.5 * Tile Size and another side > Tile Size.
 * @return Return "fold_copies" for "PartitionInfo". Refer to the definition of
 * fold_copies in "struct PartitionInfo" in Mapping.h.
 */
int Mapping::adjust_split_array(double &Split_heightArray,
                                double &Split_widthArray) {
  int fold_copies = 1;
  bool horizontal_fold = false;
  // Vertical Folding. At most one branch of "while" can be executed.
  while (Split_heightArray * 2 < Tile[0] && Split_widthArray > Tile[1]) {
    Split_heightArray *= 2;
    Split_widthArray /= 2;
    fold_copies *= 2;
    if (fold_copies >= Tile[0]) {
      break;
    } // Fold times limited by Tile Size
  }
  // Horizontal Folding.
  while (Split_widthArray * 2 < Tile[1] && Split_heightArray > Tile[0]) {
    horizontal_fold = true;
    Split_widthArray *= 2;
    Split_heightArray /= 2;
    fold_copies *= 2;
    if (fold_copies >= Tile[1]) {
      break;
    } // Fold times limited by Tile Size
  }

  // change to negative value when performing horizontal folding.
  fold_copies = horizontal_fold ? -fold_copies : fold_copies;
  return fold_copies;
}

/**
 * @brief Similar to weight_partition, but multiple Tile Size can be chosen to
 * hold the weights. The Subarray Size is fixed here, however.
 * The problem becomes an optimization problem, and we'll first define the
 * search space and then go through all possible params to find the best one.
 */
void Mapping::weight_mixed_split(PartitionInfo &info, vector<int> weight_unfold,
                                 int buffer_demand, int duplication) {
  // 1. Define Search Space for Tile Size.
  int MAX_TILE_SIZES = 4; // How many types of Tile Size are allowed;
  vector<vector<int>> tile_sizes = {{2, 2}, {2, 4}, {4, 4}, {8, 8}};
}

/**
 * @brief establish Hierarchical Tree Network-on-Chip with routers.
 */
void Mapping::Htree_NoC(HtreeNoCInfo &info, PartitionInfo part_info) {
  // We need to Calculate how many input hops and output hops are needed to
  // transfer IFMs and OFMs into/out of subarrays. And in this function, the
  // usage of all the subarrays in all the tiles of the current
  // layer is assumed to be the SAME. So we only need to investigate the input
  // hops and output hops of ONE TILE here. The modeling procedure of Htree hops
  // calculation is in "Graphs.pptx".

  // NOTE: A hop means transferring a packet (with data amount to ONE SUBARRAY)
  // through a HTree router. So macroRows/macroCols really matter.

  // At present, Tile Size and Subarray Size are all assumed to be the power of
  // 2, and the width and height are the same.
  auto Split_array = part_info.Split_array;
  auto intra_dup = part_info.Intra_tile; // Duplications inside a Tile.
  auto fold_copies = part_info.fold_copies;
  int macroRows, macroCols;
  if (fold_copies > 0) { // fold vertically
    macroRows = ceil(Split_array[0] / fold_copies) * fold_copies;
    macroCols = ceil(Split_array[1]);
  } else { // fold horizontally
    macroRows = ceil(Split_array[0]);
    macroCols = ceil(Split_array[1] / -fold_copies) * -fold_copies;
  }

  macroRows = ceil((double)macroRows / part_info.intraRowDup);
  macroCols = ceil((double)macroCols / part_info.intraColDup);

  // Hops for a packet to be transferred into/out of a subarray
  int routing_hops = log2(Tile[0]); // Tile[0] should be the power of 2;

  int input_hops, output_hops;

  int curRows = Tile[0], curCols = Tile[1];
  if (fold_copies == 1) {
    // Condition 1: Intra-Tile duplication and folding are not considered in
    // this condition. The most simple condition.
    int sub_routers = 0; // The num of routers concerned
    for (int i = 1; i <= routing_hops; i++) {
      sub_routers += ceil((double)macroCols / curCols);
      curCols /= 2;
    }
    input_hops = macroRows * sub_routers;

    sub_routers = 0;
    for (int i = 1; i <= routing_hops; i++) {
      sub_routers += ceil((double)macroRows / curRows);
      curRows /= 2;
    }
    output_hops = macroCols * sub_routers;
  } else if (fold_copies > 1) {
    // Condition 2: Intra-Tile duplication not considered in this condition.
    // Folding weight vertically.
    int curDup = 1;
    // After folding weights, we divide the weights into several groups
    // row-wise to calculate input_hops.
    int router_sum = 0; // total sum of routers for one "group" of input.
    for (int i = 1; i <= routing_hops; i++) {
      router_sum += curDup * ceil((double)macroCols / curCols);
      curCols /= 2;
      if (curDup < fold_copies) {
        curDup *= 2;
      }
    }
    input_hops = macroRows / fold_copies * router_sum;

    router_sum = 0;
    int group_rows = macroRows / fold_copies;
    for (int i = 1; i <= routing_hops; i++) {
      router_sum += ceil((double)group_rows / curRows);
      curRows /= 2;
    }
    output_hops = macroCols * fold_copies * router_sum;
  } else {
    // Condition 3: Intra-Tile duplication not considered in this condition.
    // Folding weight horizontally.
    fold_copies = -fold_copies; // The only place where fold_copies changed!!!
    int router_sum = 0;
    int group_cols = macroCols / fold_copies;
    for (int i = 1; i <= routing_hops; i++) {
      router_sum += ceil((double)group_cols / curCols);
      curCols /= 2;
    }
    input_hops = macroRows * fold_copies * router_sum;

    int used_routers; // The num of routers used counting from the column
    int divide_num = 1;
    int concerned_columns = macroCols;
    router_sum = 0;
    curCols = Tile[1];
    for (int i = routing_hops; i >= 1; i--) {
      // Search the use of routers from low levels to high levels.
      // Correspond to "yellow->blue->black" router in "Graphs.pptx".
      divide_num *= 2;
      used_routers = ceil((double)macroRows / divide_num);
      if ((1 << i) <= fold_copies) {
        concerned_columns /= 2;
      }
      router_sum += used_routers * concerned_columns;
      curCols /= 2;
    }
    output_hops = router_sum;
  }

  // Duplication are considered here. Just duplicate input_hops/output_hops.
  input_hops *= intra_dup;
  output_hops *= intra_dup;
  info.Tile_NoC.resize(2);
  info.Tile_NoC[0] = input_hops;
  info.Tile_NoC[1] = output_hops;
  info.Htree_level = routing_hops;

  if (testMode) {
    cout << "*************** Htree_NoC Test Case " << ++HTreeTestCnt
         << " ***************" << endl;
    cout << "input_hops = " << input_hops << ", output_hops = " << output_hops
         << endl;
  }
}

/**
 * @brief Calculate Buffer_read and Buffer_write. Buffer_read: the num of data
 * read from buffer; Buffer_write: the num of data write to buffer
 * 
 * @param input_vec_num: num of input vecs sent to Macros.
 * 
 * @return pair<int, int> first: Buffer_read; second: Buffer_write
 */
pair<int, int> Mapping::Calculate_Buffer(vector<double> Split_tile,
                                         int duplication, int input_vec_num) {
  int Buffer_read, Buffer_write;
  if (transform_method == COLUMN_TRANS) {
    Buffer_read = k1 * k2 * input_vec_num * pyInput->shape[1] * pyParam->a_precision *
                  Split_tile[1];
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] *
                   pyParam->a_precision;
  } else if (transform_method == MATRIX_TRANS) {
    Buffer_read = k1 * input_vec_num * pyInput->shape[1] * pyParam->a_precision *
                  Split_tile[1];
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] *
                   pyParam->a_precision;
  } else if (transform_method == ROW_TRANS) {
    Buffer_read = input_vec_num * pyInput->shape[1] * pyParam->a_precision * 
                  Split_tile[1];
    Buffer_write = 2 * k2 * input_vec_num * outChannels * Split_tile[0] *
                   pyParam->a_precision;
  } else if (transform_method == LW_MATMUL_TRANS) {
    Buffer_read = input_vec_num * pyInput->shape[0] * pyParam->a_precision * 
                  Split_tile[1]; // pyInput->shape[0] = input vec dimension
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] *
                   pyParam->a_precision;
  } else if (transform_method == RW_MATMUL_TRANS) {
    Buffer_read = input_vec_num * pyInput->shape.back() * pyParam->a_precision * 
                  Split_tile[1]; // pyInput->shape.back() = input vec dimension
    Buffer_write = 2 * input_vec_num * outChannels * Split_tile[0] *
                   pyParam->a_precision;
  } else {
    throw runtime_error("In Calculate_Buffer(), unidentified transform_method found!!!");
  }
  return pair<int, int>(Buffer_read, Buffer_write);
}

int Mapping::Calculate_TileNoC(PartitionInfo info, vector<int> Tile_NoC,
                               int duplication, int input_vec_num) {
  // Tile_NoC[0] = input_hops, Tile_NoC[1] = output_hops;
  int TileNoC_num = Tile_NoC[0] * info.Intra_tile * info.Inter_tile *
                    info.Split_array[0] * Subarray[0] * input_vec_num / duplication *
                    pyParam->a_precision;
  if (transform_method == COLUMN_TRANS || 
      transform_method == LW_MATMUL_TRANS || 
      transform_method == RW_MATMUL_TRANS) {
    TileNoC_num += Tile_NoC[1] * info.Intra_tile * info.Inter_tile *
                   info.Split_array[1] * Subarray[1] * input_vec_num / duplication *
                   pyParam->a_precision;
  } else if (transform_method == MATRIX_TRANS ||
             transform_method == ROW_TRANS) {
    TileNoC_num += Tile_NoC[1] * info.Intra_tile * info.Inter_tile *
                   info.Split_array[1] * Subarray[1] * input_vec_num / duplication /
                   k1 * pyParam->a_precision;
  }
  return TileNoC_num;
}

/**
 * @brief Calculate Performance for Tile
 *
 */
void Mapping::Calculate_TilePerformance(TilePerfInfo &tilePerfInfo,
                                        PartitionInfo partitionInfo,
                                        int Htree_level, int HtreeNoC,
                                        json htreePerf, json macroPerf,
                                        json spmPerf, int duplication, int input_vec_num) {
  double Htree_area = htreePerf["area"], Htree_energy = htreePerf["energy"],
         Htree_latency = htreePerf["latency"];
  double Macro_area = macroPerf["area"], Macro_energy = macroPerf["energy"],
         Macro_latency = macroPerf["latency"];
  double SPM_area = spmPerf["area"], SPM_read_energy = spmPerf["read_energy"],
         SPM_write_energy = spmPerf["write_energy"],
         SPM_latency = spmPerf["latency"];
  double VMM_energy = CIM_num * Macro_energy * pyWeight->weight_sparsity *
                      pyInput->input_sparsity * pyParam->a_precision;
  double SPM_energy =
      Buffer_read * SPM_read_energy + Buffer_write * SPM_write_energy;
  double HtreeNoC_energy = TileNoC_num * Htree_energy;
  tilePerfInfo.energy = VMM_energy + SPM_energy + HtreeNoC_energy;

  vector<double> Data(2);
  Data[0] = MIN(Subarray[0], partitionInfo.Split_array[0] * Subarray[0]);
  Data[1] = MIN(Subarray[1], partitionInfo.Split_array[1] * Subarray[1]);
  partitionInfo.Split_array[0] = ceil(partitionInfo.Split_array[0]);
  partitionInfo.Split_array[1] = ceil(partitionInfo.Split_array[1]);

  if (transform_method == MATRIX_TRANS || transform_method == ROW_TRANS) {
    Data[1] = Subarray[1] / k1;
  }

  double HtreeNoC_latency = (Htree_level + (Data[0] - HtreeNoC) / HtreeNoC) *
                            (partitionInfo.Split_array[1] + 1) * Htree_latency;
  double Buffer_latency =
      (Data[0] + partitionInfo.Split_array[1] * Data[1]) * SPM_latency;
  if ((partitionInfo.Intra_tile * partitionInfo.Split_array[0] - 1) *
          (HtreeNoC_latency + Buffer_latency) >
      Macro_latency) {
    tilePerfInfo.latency = partitionInfo.Intra_tile *
                           partitionInfo.Split_array[0] *
                           (HtreeNoC_latency + Buffer_latency);
    // if (pyParam->layer == 9) {
    //   cout << "In layer 9, latency = " << tilePerfInfo.latency << endl;
    // }
  } else {
    // cout << "Going here!!" << endl;
    tilePerfInfo.latency = HtreeNoC_latency + Buffer_latency + Macro_latency;
    // if (pyParam->layer == 9) {
    //   cout << "In layer 9, latency = " << tilePerfInfo.latency << endl;
    // }
  }
  
  tilePerfInfo.latency *= input_vec_num / duplication * pyParam->a_precision;
  // tilePerfInfo.latency *= pyInput->shape[2] * pyInput->shape[3] / duplication *
  //                         pyParam->a_precision;

  tilePerfInfo.area = partitionInfo.Inter_tile *
                      (SPM_area + Htree_area + Macro_area * Tile[0] * Tile[1]);
}

/**
 * @brief Calculate ip_index. The format of each unit is: [src, dst, packet]
 *
 * @return vector<vector<int>>
 */
vector<vector<int>> Mapping::Mesh_NoC() {
  vector<vector<int>> traffic;
  // When Mesh_NoC is called, "mapping.txt" is fully calculated;
  // "meshconnect.txt" also implies the correlation between layers. We need to
  // replace "prelayer_tile" and "nextlayer_tile" in "meshconnect.txt", which
  // are set to (0,0) currently.

  /*** 1: Parse the Info in mapping.txt in advance ***/
  ifstream f_map(PATH_MAPPING);
  // record the range of virtual tiles used in each layer. Each item in
  // mapping_vTiles follows the format: [tile_start, tile_end].
  vector<vector<int>> mapping_vTiles;
  vector<int> duplication;
  vector<int> tile_rows; // The rows occupied by each layer;
  vector<int> tile_cols; // The columns occupied by each layer;
  string line;
  int cur_tile = 0;

  while (getline(f_map, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    vector<int> Split_tile(2);
    Split_tile[0] = stoi(tokens[3]), Split_tile[1] = stoi(tokens[5]);
    int tile_start = cur_tile,
        tile_end = cur_tile + Split_tile[0] * Split_tile[1] - 1;
    mapping_vTiles.push_back({tile_start, tile_end});
    duplication.push_back(stoi(tokens[15]));
    tile_rows.push_back(Split_tile[0]);
    tile_cols.push_back(Split_tile[1]);
    cur_tile += Split_tile[0] * Split_tile[1];
  }

  /*** 2: Parse the Info in meshconnect.txt ***/
  ifstream f_mesh(PATH_MESHCONNECT);
  string content;
  while (getline(f_mesh, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    int cur_layer = stoi(tokens[1]), next_layer = stoi(tokens[3]);
    int total_volumn = stoi(tokens[7]);
    int tile_start, tile_end;
    /*** 3: Write prelayer_tile and nextlayer_tile into file ***/
    if (line.find("prelayer_tile: 0 0") != string::npos) {
      tile_start = mapping_vTiles[cur_layer - 1][0];
      tile_end = mapping_vTiles[cur_layer - 1][1];
      line = line.replace(line.find("prelayer_tile: 0 0"), 18,
                          "prelayer_tile: " + to_string(tile_start) + " " +
                              to_string(tile_end));
    }
    if (line.find("nextlayer_tile: 0 0") != string::npos) {
      tile_start = mapping_vTiles[next_layer - 1][0];
      tile_end = mapping_vTiles[next_layer - 1][1];
      line = line.replace(line.find("nextlayer_tile: 0 0"), 19,
                          "nextlayer_tile: " + to_string(tile_start) + " " +
                              to_string(tile_end));
    }
    content += line + '\n';
    /*** 4: Deciding "traffic" for connected layers ***/
    // Partsum and concat operations can be done in routers. We decide that
    // these operations are performed in the "first" router (with the minimum
    // virtual tile id) of the layer to be passed packets. Other routers only
    // need to store and forward packets as normal routers do.
    tile_start = mapping_vTiles[cur_layer - 1][0];
    tile_end = mapping_vTiles[cur_layer - 1][1];
    int first_router_tile = mapping_vTiles[next_layer - 1][0];
    // (i) Packets from cur_layer are sent to the "first" router of next_layer
    for (int i = tile_start; i <= tile_end; i++) {
      // TODO: simplified packet calculation. What if A tile is not totally
      // occupied by unfolded weight?

      // outputs of tiles in a row are concatenated to form the whole output.
      int packet =
          total_volumn / tile_cols[cur_layer - 1] * pyParam->a_precision;
      traffic.push_back({i, first_router_tile, packet});
    }

    tile_start = mapping_vTiles[next_layer - 1][0];
    tile_end = mapping_vTiles[next_layer - 1][1];
    // (ii) Packets from cur_layer are sent to the "first" router of next_layer
    for (int i = tile_start + 1; i <= tile_end; i++) {
      int packet =
          total_volumn / tile_rows[next_layer - 1] * pyParam->a_precision;
      traffic.push_back({tile_start, i, packet});
    }
  }
  f_mesh.close();
  ofstream f_mesh_out(PATH_MESHCONNECT);
  f_mesh_out << content;
  f_mesh_out.close();
  ofstream of_traffic(PATH_TRAFFIC);
  for (auto vec : traffic) {
    for (auto id : vec) {
      of_traffic << id << " ";
    }
    of_traffic << endl;
  }
  of_traffic.close();

  // Pass the Tile num
  ofstream f_tile(PATH_TILENUM);
  f_tile << cur_tile;
  f_tile.close();
  return traffic;
}

/**
 * @brief Read MappingInfo from mapping.txt, to be validated.
 *
 */
void Mapping::readMappingInfo(ifstream &f, MappingInfo &mappingInfo,
                              LayerInfo layerInfo) {
  string ll;
  while (getline(f, ll)) {
    istringstream iss(ll);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    int layer = stoi(tokens[1]);
    if (layerInfo.prelayer == layer) {
      mappingInfo.tile_rows[0] = stod(tokens[3]);
      mappingInfo.tile_cols[0] = stod(tokens[5]);
      mappingInfo.Intra_tile[0] = stod(tokens[11]);
      mappingInfo.tile_nums[0] = stod(tokens[13]);
      mappingInfo.duplication[0] = stod(tokens[15]);
    }
    if (layerInfo.nextlayer == layer) {
      mappingInfo.tile_rows[1] = stod(tokens[3]);
      mappingInfo.tile_cols[1] = stod(tokens[5]);
      mappingInfo.Intra_tile[1] = stod(tokens[11]);
      mappingInfo.tile_nums[1] = stod(tokens[13]);
      mappingInfo.duplication[1] = stod(tokens[15]);
    }
  }
}

/**
 * @brief Write performance and tile mapping info into files.
 *
 */
void Mapping::writeInfo(double area, double latency, double energy,
                        string mapOutPath, int total_layer) {
  ofstream of(mapOutPath, ios::app);
  of << setprecision(8);
  ifstream f_mesh(PATH_MESHCONNECT);

  set<int> layer_recorded;
  set<string> cnn_module;
  int residual_layers = 0;

  string line, last_type;
  while (getline(f_mesh, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    int layer = stoi(tokens[1]);
    if (layer_recorded.find(layer) == layer_recorded.end()) {
      layer_recorded.insert(layer);
      string type = tokens[5];
      bool change = false;
      if (type == "residual_conv1*1") {
        residual_layers++;
        // cout << "Find residual_conv1, layer = " << layer << endl;
      }
      if (type != "fc") {
        last_type = type; // record the last type before fc layers.
      }
      if (cnn_module.find(type) == cnn_module.end()) {
        cnn_module.insert(type);
        // TODO: another patch for correct output of fc layer name.
        if (type == "fc") {
          // For the first fc, usually the previous layer should be a conv(or
          // other types), and the next following layers should all be fc
          // layers.
          type = last_type;
          change = true;
        }
      }
      // cout << "layer = " << layer << " type = " << type << endl;
      int tile_1 = stoi(tokens[9]); // prelayer
      int tile_2 = stoi(tokens[10]);
      // cout << "Before writeTileInfo, Layer = " << layer << endl;
      writeTileInfo(tile_1, tile_2, layer, type, residual_layers, of);
      // cout << "After writeTileInfo!!!" << endl;

      if (stoi(tokens[3]) == total_layer) {
        // nextlayer
        tile_1 = stoi(tokens[12]);
        tile_2 = stoi(tokens[13]);
        if (change) {
          type = "fc";
        }
        writeTileInfo(tile_1, tile_2, total_layer, type, residual_layers, of);
      }
    }
  }
  of << "total energy (mJ): " << energy * 1000
     << " total latency (ms): " << latency * 1000
     << " total area (mm2): " << area << endl
     << endl;

  // Append placing.txt to the back of mapping_tid.txt
  of << "##########################  mapping relationship between virtual "
        "tiles and physical tiles ##########################"
     << endl
     << endl;
  ifstream f_placing(PATH_PLACING);
  while (getline(f_placing, line)) {
    of << line << endl;
  }
  of.close();
}

/**
 * @brief Called by writeInfo. Actually write the tile info into
 * mapping_tid.txt.
 *
 */
void Mapping::writeTileInfo(int tile_1, int tile_2, int layer, string type,
                            int residual_layers, ofstream &of) {
  ifstream f(PATH_PERFORMANCE); // merge performance.txt into mapping_tid.txt
  string line;
  while (getline(f, line)) {
    istringstream iss(line);
    vector<string> tokens;
    string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
    // Filter the "total energy" line
    if (tokens.size() != 8 || tokens[0] != "layer:")
      continue;
    if (layer == stoi(tokens[1])) {
      double energy = stod(tokens[3]) * 1000;  // mJ
      double latency = stod(tokens[5]) * 1000; // ms
      double area = stod(tokens[7]);           // mm^2
      if (type == "residual_conv1*1") {
        // cout << "In writeTileInfo, residual_conv1*1, layer = " << layer <<
        // endl;
        of << "layer: residual"
           << "; type: " << type << "; tiles: " << tile_1 << "-" << tile_2
           << "; energy (mJ): " << energy << "; latency (ms): " << latency
           << "; area (mm2): " << area << endl;
      } else {
        of << "layer: " << layer - residual_layers << "; type: " << type
           << "; tiles: " << tile_1 << "-" << tile_2
           << "; energy (mJ): " << energy << "; latency (ms): " << latency
           << "; area (mm2): " << area << endl;
      }
    }
  }
}

/**
 * @brief test weight partition using layers from real network. Multiple
 * duplication conditions will be considered in our test.
 *
 */
void Mapping::test_partition() {
  testMode = true;
  cout << "test_partition!!!" << endl;
  // Pre-set some parameters in advance.
  Tile.resize(2);
  Subarray.resize(2);
  Tile[0] = 2, Tile[1] = 2;
  Subarray[0] = 512, Subarray[1] = 512;
  bufferSizeTile = 128 * 1024 * 8; // 1KB
  PartitionInfo info;
  vector<int> weight_unfold(2);
  double buffer_demand;
  // vector<int> dup = {64, 16, 4, 4, 1, 1, 1, 1, 1, 1, 1}; // For Hybrid
  vector<int> dup = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // For No Dup
  // vector<int> dup = {256, 64, 16, 16, 4, 4, 1, 1, 1, 1, 1}; // For Extreme

  std::ofstream file(PATH_TEST_PARTITION, std::ios::trunc);
  file.close();

  // Using VGG-11 running on CIFAR-10 as example.
  // For Layer 1-11, assign weight_unfold and corresponding buffer_demand.
  // Duplication: we consider three types of pipeline here, and the duplication
  // nums are assigned above.

  weight_unfold[0] = 27, weight_unfold[1] = 512; // Layer 1
  buffer_demand = 1005;
  weight_partition(info, weight_unfold, buffer_demand, dup[0]);

  weight_unfold[0] = 192, weight_unfold[1] = 1920; // Layer 2
  buffer_demand = 10560;
  weight_partition(info, weight_unfold, buffer_demand, dup[1]);

  weight_unfold[0] = 384, weight_unfold[1] = 3840; // Layer 3
  buffer_demand = 10880;
  weight_partition(info, weight_unfold, buffer_demand, dup[2]);

  weight_unfold[0] = 768, weight_unfold[1] = 3840; // Layer 4
  buffer_demand = 21760;
  weight_partition(info, weight_unfold, buffer_demand, dup[3]);

  weight_unfold[0] = 768, weight_unfold[1] = 7680; // Layer 5
  buffer_demand = 11520;
  weight_partition(info, weight_unfold, buffer_demand, dup[4]);

  weight_unfold[0] = 1536, weight_unfold[1] = 7680; // Layer 6
  buffer_demand = 23040;
  weight_partition(info, weight_unfold, buffer_demand, dup[5]);

  weight_unfold[0] = 1536, weight_unfold[1] = 7680; // Layer 7
  buffer_demand = 12800;
  weight_partition(info, weight_unfold, buffer_demand, dup[6]);

  weight_unfold[0] = 1536, weight_unfold[1] = 7680; // Layer 8
  buffer_demand = 12800;
  weight_partition(info, weight_unfold, buffer_demand, dup[7]);

  weight_unfold[0] = 512, weight_unfold[1] = 512; // Layer 9
  buffer_demand = 2560;
  weight_partition(info, weight_unfold, buffer_demand, dup[8]);

  weight_unfold[0] = 512, weight_unfold[1] = 512; // Layer 10
  buffer_demand = 2560;
  weight_partition(info, weight_unfold, buffer_demand, dup[9]);

  weight_unfold[0] = 512, weight_unfold[1] = 10; // Layer 11
  buffer_demand = 2560;
  weight_partition(info, weight_unfold, buffer_demand, dup[10]);
}

/**
 * @brief test function Htree_NoC(). Test whether "input hops" and "output hops"
 * are calculated correctly in different situations.
 *
 */
void Mapping::test_HtreeHops() {
  testMode = true;
  HtreeNoCInfo info;
  PartitionInfo part_info;

  // Different sets of part_info. Split_array, Intra_tile and fold_copies are
  // required to call Htree_NoC(). Tile Size also needs to be specified.
  Tile.resize(2);
  Tile[0] = 8, Tile[1] = 8;
  part_info.Split_array.resize(2);

  // Examples of Htree_NoC calculation in "Graphs.pptx".
  part_info.Split_array[0] = 7.5, part_info.Split_array[1] = 5.5;
  part_info.Intra_tile = 1;
  part_info.fold_copies = 1;
  Htree_NoC(info, part_info); // 1:PPT Slide 1, no folding

  part_info.Split_array[0] = 8, part_info.Split_array[1] = 6;
  part_info.Intra_tile = 1;
  part_info.fold_copies = 2;
  Htree_NoC(info, part_info); // 2:PPT Slide 1, vertically folding once

  part_info.Split_array[0] = 5.75, part_info.Split_array[1] = 5.8;
  part_info.Intra_tile = 1;
  part_info.fold_copies = 4;
  Htree_NoC(info, part_info); // 3:PPT Slide 1, vertically folding twice

  part_info.Split_array[0] = 4.5, part_info.Split_array[1] = 5.7;
  part_info.Intra_tile = 1;
  part_info.fold_copies = 8;
  Htree_NoC(info, part_info); // 4:PPT Slide 1, vertically folding three times

  part_info.Split_array[0] = 6, part_info.Split_array[1] = 5.7;
  part_info.Intra_tile = 1;
  part_info.fold_copies = 2;
  Htree_NoC(info, part_info); // 5:PPT Slide 2, vertically folding once

  part_info.Split_array[0] = 7, part_info.Split_array[1] = 6;
  part_info.Intra_tile = 1;
  part_info.fold_copies = -2;
  Htree_NoC(info, part_info); // 6:PPT Slide 4, horizontally folding once

  part_info.Split_array[0] = 6.5, part_info.Split_array[1] = 7.8;
  part_info.Intra_tile = 1;
  part_info.fold_copies = -2;
  Htree_NoC(info, part_info); // 7:PPT Slide 5, horizontally folding once

  part_info.Split_array[0] = 6.5, part_info.Split_array[1] = 7.8;
  part_info.Intra_tile = 1;
  part_info.fold_copies = -4;
  Htree_NoC(info, part_info); // 8:PPT Slide 5, horizontally folding twice

  part_info.Split_array[0] = 6.5, part_info.Split_array[1] = 7.8;
  part_info.Intra_tile = 1;
  part_info.fold_copies = -8;
  Htree_NoC(info, part_info); // 9:PPT Slide 5, horizontally folding three times

  part_info.Split_array[0] = 8, part_info.Split_array[1] = 8;
  part_info.Intra_tile = 4; // Intra duplication = 4;
  part_info.fold_copies = -2;
  part_info.intraRowDup = 2, part_info.intraColDup = 2;
  Htree_NoC(info, part_info); // 10:PPT Slide 5, horizontally folding once

  part_info.Split_array[0] = 4, part_info.Split_array[1] = 4;
  part_info.Intra_tile = 1; // Intra duplication = 4;
  part_info.fold_copies = -2;
  part_info.intraRowDup = 1, part_info.intraColDup = 1;
  Htree_NoC(info, part_info); // 11:PPT Slide 5, horizontally folding once
}

} // namespace Refactor
