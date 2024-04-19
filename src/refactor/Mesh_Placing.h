/**
 * @file Mesh_Placing.h
 * @author booniebears
 * @brief
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MESH_PLACING_H_
#define MESH_PLACING_H_

#include <map>
#include <vector>

#include "defines.h"
using namespace std;

namespace Refactor {

struct MeshInfo {
  double link_length = 0;
  double Mesh_latency = 0;
};

struct TileLocation {
  int x;
  int y;

  bool operator==(const TileLocation &loc) const {
    return x == loc.x && y == loc.y;
  }

  bool operator!=(const TileLocation &loc) const {
    return x != loc.x || y != loc.y;
  }

  bool operator<(const TileLocation &loc) const {
    if (x != loc.x) {
      return x < loc.x;
    } else {
      return y < loc.y;
    }
  }
};

struct TimeSpan {
  int start = 0;
  int end = 0;
};

class Mesh_Placing {
public:
  Mesh_Placing(int filter, int bandWidth, int totalTiles, double Mesh_latency);
  virtual ~Mesh_Placing() {}

  MeshInfo Mesh_mapping_random_pipeline(map<pair<int, int>, int> traffic_mp);
  MeshInfo Mesh_mapping_energy_pipeline(map<pair<int, int>, int> traffic_mp);

  // Test of Mesh_Placing
  void test_schedule_idle();
  void test_NMAP();

private:
  MeshInfo random_pipeline(vector<vector<int>> ip_index,
                           vector<TileLocation> tileLocation);
  MeshInfo NMAP_pipeline(vector<vector<int>> ip_index,
                         vector<TileLocation> tileLocation);

  // Algorithm for Placing tiles physically
  void NMAP(map<pair<int, int>, int> traffic_mp,
            vector<TileLocation> &tileLocation);
  double schedule_idle(int x_to, int y_to, int x_from, int y_from,
                       double send_latency, double transfer_latency);
  double schedule_idle_simplified(int x_to, int y_to, int x_from, int y_from,
                                  double send_latency, double transfer_latency);

  int filter, bandWidth;
  int totalTiles;
  double per_router_latency;
  // Location -> Last time the router is used.
  map<TileLocation, TimeSpan> timeTable;
};

} // namespace Refactor

#endif // !MESH_PLACING_H_