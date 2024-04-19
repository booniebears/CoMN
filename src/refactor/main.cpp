/**
 * @file main.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <string>

#include "Mapping.h"
#include "Mesh_Placing.h"
#include "Perf_Evaluator.h"
#include "PyActInput.h"
#include "PyInput.h"
#include "PyParam.h"
#include "PyWeight.h"
#include "defines.h"
#include "json.hpp"

using namespace Refactor;
using namespace std;
using json = nlohmann::json;

bool do_mapping = false;
bool do_activation = false;
bool do_pipeline = false;
bool do_mesh = false;
bool do_PPA = false;
bool do_test = false;
string user_name = "";
string tid = "";

bool parse_arg(int argc, char **argv);

int main(int argc, char **argv) {
  if (!parse_arg(argc, argv)) {
    return 0;
  }

  PyParam *pyParam = new PyParam();
  PyWeight *pyWeight = new PyWeight();
  PyInput *pyInput = new PyInput();
  PyActInput *pyActInput = new PyActInput();

  Mapping *mapping = new Mapping(pyParam, pyWeight, pyInput, pyActInput);
  if (do_pipeline) {
    cout << "pipeline_optimized!!" << endl;
    mapping->pipeline_optimized();
  }
  if (do_PPA) {
    cout << "PPA_cost!!" << endl;
    PPA_cost();
  }
  if (do_mesh) {
    cout << "Mesh_operation!!" << endl;
    // Figure out traffic of NoC first.
    mapping->Mesh_NoC();
    mapping->Mesh_operation(user_name, tid);
  }
  if (do_mapping) {
    // cout << "mapping_modules!!!" << endl;
    mapping->mapping_modules();
  }
  if (do_activation) {
    // cout << "activation_modules!!!" << endl;
    mapping->activation_modules();
  }
  if (do_test) {
    cout << "*******************Start Unit tests!!!*******************" << endl;
    cout << "Testing Items are listed below." << endl;
    vector<string> testItems = {"test_schedule_idle", "test_partition",
                                "test_HtreeHops", "test_NMAP"};
    for (int i = 0; i < testItems.size(); i++) {
      cout << "Item " << i + 1 << " , " << testItems[i] << endl;
    }
    vector<int> chosenItems = {0}; // chosen item ids in "testItems"
    cout << "Chosen Test: " << testItems[0] << endl;
    for (auto item : chosenItems) {
      if (item == 0) {
        Mesh_Placing *placing = new Mesh_Placing(1, 1, 1, 1);
        placing->test_schedule_idle();
        free(placing);
      }
      if (item == 1) {
        mapping->test_partition();
      }
      if (item == 2) {
        mapping->test_HtreeHops();
      }
      if (item == 3) {
        Mesh_Placing *placing = new Mesh_Placing(1, 1, 1, 1);
        placing->test_NMAP();
        free(placing);
      }
    }
  }

  free(pyParam);
  free(pyWeight);
  free(pyInput);
  free(pyActInput);
  free(mapping);
  return 0;
}

bool parse_arg(int argc, char **argv) {
  if (argc == 4) {
    if (strcmp(argv[1], "--activation_modules") == 0) {
      do_activation = true;
    } else if (strcmp(argv[1], "--mapping_modules") == 0) {
      do_mapping = true;
    } else if (strcmp(argv[1], "--pipeline_optimized") == 0) {
      do_pipeline = true;
    } else if (strcmp(argv[1], "--Mesh_operation") == 0) {
      do_mesh = true;
    } else if (strcmp(argv[1], "--PPA_cost") == 0) {
      do_PPA = true;
    } else if (strcmp(argv[1], "--test_modules") == 0) {
      do_test = true;
    } else {
      cout << "[CoMN_refactor] Method not supported here!!!" << endl;
      return false;
    }
    user_name = argv[2];
    tid = argv[3];
    return true;
  } else {
    cout << "Arg format is not correct!!!" << endl;
    return false;
  }
}
