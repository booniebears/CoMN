/**
 * @file Mesh_Placing.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <math.h>
#include <set>

#include "Mesh_Placing.h"
#include "defines.h"

namespace Refactor {

Mesh_Placing::Mesh_Placing(int filter, int bandWidth, int totalTiles,
                           double Mesh_latency)
    : filter(filter), bandWidth(bandWidth), totalTiles(totalTiles),
      per_router_latency(Mesh_latency) {
  //
}

/**
 * @brief task mapping using random mapping algorithm.
 *
 */
MeshInfo Mesh_Placing::Mesh_mapping_random_pipeline(
    map<pair<int, int>, int> traffic_mp) {
  vector<TileLocation> tileLocation;
  tileLocation.resize(totalTiles);
  int tile_rows = floor(sqrt(totalTiles));
  int tile_cols = ceil((double)totalTiles / tile_rows);
  int x = 0, y = 0;
  for (int i = 0; i < totalTiles; i++) {
    tileLocation[i] = {x, y};
    y++;
    if (y >= tile_cols) {
      y = 0;
      x++;
    }
  }

  ifstream f_traffic(PATH_TRAFFIC);
  int src, dst, packet;
  vector<vector<int>> traffic;
  while (f_traffic >> src >> dst >> packet) {
    traffic.push_back({src, dst, packet});
  }
  cout << "Start executing random_pipeline!!!" << endl;
  MeshInfo meshInfo = random_pipeline(traffic, tileLocation);
  cout << "random_pipeline finished!!!" << endl;

  ofstream f_place(PATH_PLACING, ios::app);
  f_place << "Total Rows: " << tile_rows << endl;
  f_place << "Total Columns: " << tile_cols << endl;
  for (int i = 0; i < totalTiles; i++) {
    f_place << "tile: " << i << " location: [" << tileLocation[i].x << ", "
            << tileLocation[i].y << "]" << endl;
  }
  return meshInfo;
}

/**
 * @brief task mapping using nearliest mapping algorithm.
 *
 */
MeshInfo Mesh_Placing::Mesh_mapping_energy_pipeline(
    map<pair<int, int>, int> traffic_mp) {
  vector<int> mapped_task;
  vector<vector<int>> located_router;
  vector<TileLocation> tileLocation; // virtual tile id -> location on chip
  tileLocation.resize(totalTiles);
  NMAP(traffic_mp, tileLocation);
  ifstream f_traffic(PATH_TRAFFIC);
  int src, dst, packet;
  vector<vector<int>> traffic;
  while (f_traffic >> src >> dst >> packet) {
    traffic.push_back({src, dst, packet});
  }
  auto meshInfo = NMAP_pipeline(traffic, tileLocation);
  int tile_rows = floor(sqrt(totalTiles));
  int tile_cols = ceil((double)totalTiles / tile_rows);
  ofstream f_place(PATH_PLACING, ios::app);
  f_place << "Total Rows: " << tile_rows << endl;
  f_place << "Total Columns: " << tile_cols << endl;

  // TODO: Print a Grid for Physical Tile Location Mapping.
  for (int i = 0; i < totalTiles; i++) {
    f_place << "tile: " << i << " location: [" << tileLocation[i].x << ", "
            << tileLocation[i].y << "]" << endl;
  }
  return meshInfo;
}

/**
 * @brief
 *
 */
MeshInfo Mesh_Placing::random_pipeline(vector<vector<int>> ip_index,
                                       vector<TileLocation> tileLocation) {
  double link_length = 0, Mesh_latency = 0;
  auto unscheduled_traffic = ip_index;

  for (auto it = unscheduled_traffic.begin(); it != unscheduled_traffic.end();
       it++) {
    auto vec = *it;
    int src = vec[0], dst = vec[1], packet = vec[2];
    int x_from = tileLocation[src].x, y_from = tileLocation[src].y,
        x_to = tileLocation[dst].x, y_to = tileLocation[dst].y;

    unscheduled_traffic.erase(it);
    it--;

    int routers = abs(x_to - x_from) + abs(y_to - y_from) + 1;
    // For the whole packet to be sent onto link;
    double send_latency = 1.0 * packet / bandWidth;
    // For a router to transfer a filter;
    double transfer_latency = 1.0 * filter / bandWidth;
    double latency = schedule_idle(x_to, y_to, x_from, y_from, send_latency,
                                   transfer_latency);
    latency *= per_router_latency;
    // Figure out the longest latency among all the routing traffic.
    if (Mesh_latency < latency) {
      Mesh_latency = latency;
    }
    link_length += routers * packet;
    // if (unscheduled_traffic.size() % 50 == 0) {
    //   cout << "unscheduled_traffic.size = " << unscheduled_traffic.size()
    //        << endl;
    // }
  }

  return MeshInfo{link_length, Mesh_latency};
}

MeshInfo Mesh_Placing::NMAP_pipeline(vector<vector<int>> ip_index,
                                     vector<TileLocation> tileLocation) {
  // link_length: The number of store-and-forward times of Mesh routers;
  // Mesh_latency: total on-chip latency after scheduling the routing traffic;
  double link_length = 0, Mesh_latency = 0;
  vector<vector<int>> unscheduled_traffic = ip_index;
  // auto unscheduled_traffic = traffic_mp;
  set<int> mapped_tasks;
  for (int i = 0; i < totalTiles; i++) {
    mapped_tasks.insert(i);
  }
  set<int> unscheduled_tasks = mapped_tasks;
  // ofstream f_test("../test/mesh_test.txt");
  while (!unscheduled_traffic.empty()) {
    // The tile to be removed from unscheduled_tasks. Also the destination tile
    // of next scheduling traffic.
    int target_task = 0;

    /*** 1. find the virtual tile id with greatest to-destination delay in the
     *** rest of traffic that has not been scheduled.***/
    double max_latency = -1;
    for (int task : unscheduled_tasks) {
      double routing_latency = 0;
      for (auto vec : unscheduled_traffic) {
        int src = vec[0], dst = vec[1], packet = vec[2];
        if (task == dst) {
          int x_from = tileLocation[src].x, y_from = tileLocation[src].y,
              x_to = tileLocation[task].x, y_to = tileLocation[task].y;
          // The num of routers used to transfer packets
          int routers = abs(x_to - x_from) + abs(y_to - y_from) + 1;
          routing_latency +=
              (1.0 * packet / bandWidth + 1.0 * filter / bandWidth * routers) *
              per_router_latency;
          // cout << "Calculating routing_latency, dst = " << dst << endl;
        }
      }
      if (max_latency < routing_latency) {
        max_latency = routing_latency;
        target_task = task;
        // f_test << "task = " << task << endl;
        // f_test << "max_latency = " << max_latency << endl;
      }
    }

    /*** 2. Iteratively schedule traffic where "target_task" is the destination.
     *** After scheduling, traffic is removed from unscheduled_traffic.***/
    unscheduled_tasks.erase(target_task);
    // f_test << "target_task = " << target_task << endl;
    // f_test << "unscheduled_tasks.size = " << unscheduled_tasks.size() <<
    // endl; f_test << "unscheduled_traffic.size = " <<
    // unscheduled_traffic.size()
    //        << endl;

    double latency;

    for (auto it = unscheduled_traffic.begin(); it != unscheduled_traffic.end();
         it++) {
      auto vec = *it;
      int src = vec[0], dst = vec[1], packet = vec[2];
      if (target_task == dst) {
        int x_from = tileLocation[src].x, y_from = tileLocation[src].y,
            x_to = tileLocation[target_task].x,
            y_to = tileLocation[target_task].y;

        unscheduled_traffic.erase(it);
        it--; // vector traverse method. do not work for map.

        // it = unscheduled_traffic.erase(it); // map traverse method.

        int routers = abs(x_to - x_from) + abs(y_to - y_from) + 1;
        // For the whole packet to be sent onto link;
        double send_latency = 1.0 * packet / bandWidth;
        // For a router to transfer a filter;
        double transfer_latency = 1.0 * filter / bandWidth;
        latency = schedule_idle(x_to, y_to, x_from, y_from, send_latency,
                                transfer_latency);
        latency *= per_router_latency;
        // Figure out the longest latency among all the routing traffic.
        if (Mesh_latency < latency) {
          Mesh_latency = latency;
        }
        link_length += routers * packet;
      }
    }
    // cout << "unscheduled_traffic.size  = " << unscheduled_traffic.size() <<
    // endl;
  }
  cout << "link_length = " << link_length << ", Mesh_latency = " << Mesh_latency
       << endl;
  return MeshInfo{link_length, Mesh_latency};
}

/**
 * @brief Greedy Algorithm NMAP for Calculating Placing tiles physically.
 * TODO: Time costing!!!
 */
void Mesh_Placing::NMAP(map<pair<int, int>, int> traffic_mp,
                        vector<TileLocation> &tileLocation) {
  // For a given virtual tile id, return the physical tile location.
  // Suppose the chip has infinite area, and tiles are expected to be allocated
  // physically in a "RECTANGULAR" manner.
  set<int> unmapped_tasks; // unmapped set of virtual tiles(id stored in it)
  set<int> mapped_tasks;
  set<TileLocation> unallocated_tiles; // tileLocations not allocated yet
  set<TileLocation> allocated_tiles;
  cout << "totalTiles = " << totalTiles << endl;
  for (int i = 0; i < totalTiles; i++) {
    unmapped_tasks.insert(i);
  }

  int tile_rows = floor(sqrt(totalTiles));
  int tile_cols = ceil((double)totalTiles / tile_rows);
  cout << "tile_rows = " << tile_rows << endl;
  cout << "tile_cols = " << tile_cols << endl;
  for (int i = 0; i < tile_rows; i++) {
    for (int j = 0; j < tile_cols; j++) {
      unallocated_tiles.insert({i, j});
    }
  }

  /*** 1: Select tile with the greatest traffic, and place it in the center
   *** of the rectangle. ***/
  vector<int> packet_transmission(totalTiles, 0);
  for (auto &pr : traffic_mp) {
    auto key = pr.first;
    int src = key.first, dst = key.second, packet = pr.second;
    packet_transmission[src] += packet;
    packet_transmission[dst] += packet;
  }
  int max_traffic_tile = 0, max_traffic = -1;
  for (int i = 0; i < totalTiles; i++) {
    if (max_traffic < packet_transmission[i]) {
      max_traffic = packet_transmission[i];
      max_traffic_tile = i;
    }
  }
  // cout << "max_traffic_tile = " << max_traffic_tile << endl;

  unmapped_tasks.erase(max_traffic_tile);
  mapped_tasks.insert(max_traffic_tile);
  unallocated_tiles.erase({tile_rows / 2, tile_cols / 2});
  allocated_tiles.insert({tile_rows / 2, tile_cols / 2});
  tileLocation[max_traffic_tile] = {tile_rows / 2, tile_cols / 2};

  /*** 2: Iteratively select tile with the next greatest traffic from tiles
   *** already mapped onto chip physically, and place it on
   *** the location with the minimum communication(∑distance * traffic)
   *** to the same tiles. ***/
  while (!unmapped_tasks.empty()) {
    //
    int min_communication = INF;
    int max_traffic = 0;
    int max_unmapped_task = 0; // unmapped task with greatest traffic

    map<pair<int, int>, int> mp; // Traffic between unmapped and mapped task.
    map<pair<int, int>, int> tmp_mp;
    // Max Traffic with mapped_tasks
    for (int task : unmapped_tasks) {
      int traffic = 0;
      tmp_mp.clear();
      for (int mapped_task : mapped_tasks) {
        tmp_mp[{task, mapped_task}] +=
            traffic_mp[{task, mapped_task}] + traffic_mp[{mapped_task, task}];
        traffic +=
            traffic_mp[{task, mapped_task}] + traffic_mp[{mapped_task, task}];
        // for (auto vec : ip_index) {
        //   int src = vec[0], dst = vec[1], packet = vec[2];
        //   if (src == task && dst == mapped_task ||
        //       src == mapped_task && dst == task) {
        //     traffic += packet;
        //     tmp_mp[{task, mapped_task}] += packet;
        //   }
        // }
      }
      if (traffic > max_traffic) {
        mp = tmp_mp;
        max_traffic = traffic;
        max_unmapped_task = task;
      }
    }
    // cout << "max_traffic = " << max_traffic << endl;
    // cout << "max_unmapped_task = " << max_unmapped_task << endl;

    // Min Communication
    TileLocation bestLocation;
    for (auto loc : unallocated_tiles) {
      double communication = 0;
      for (int mapped_task : mapped_tasks) {
        double total_packet = 0;
        TileLocation mapped_loc = tileLocation[mapped_task];
        double manhatten_dis =
            abs(mapped_loc.x - loc.x) + abs(mapped_loc.y - loc.y);
        total_packet = mp[{max_unmapped_task, mapped_task}];
        communication += total_packet * manhatten_dis;
      }
      if (communication < min_communication) {
        min_communication = communication;
        bestLocation = loc;
      }
    }

    unmapped_tasks.erase(max_unmapped_task);
    mapped_tasks.insert(max_unmapped_task);
    unallocated_tiles.erase(bestLocation);
    allocated_tiles.insert(bestLocation);
    tileLocation[max_unmapped_task] = bestLocation;
  }

  cout << "NMAP Greedy Mapping Finished!!!" << endl;
}

/**
 * @brief Calculate the on-chip latency of the whole NoC. Traffic congestion in
 * routing are also considered in this method.
 *
 * @param transfer_latency The ideal routing latency of a "filter" between two
 * adjacent routers without considering any congestion. Also, this "latency" has
 * not been multiplied by per_router_latency.
 * @param send_latency The latency for a whole packet to be sent onto link.
 */
double Mesh_Placing::schedule_idle(int x_to, int y_to, int x_from, int y_from,
                                   double send_latency,
                                   double transfer_latency) {
  // The final On-chip latency returned after scheduling is supposed to be
  // greater than the "latency" without considering any congestion.

  // Simplified Assumption One: One router can only handle one forward request
  // at one time; send and receive functions are incompatible. Subsequent
  // forwarding requests need to wait for the completion of the previous
  // forwarding request.

  // Simplified Assumption Two: All routers employ XY routing, i.e., packets are
  // first transferred in the "row" direction, and then tranferred in the
  // "column" direction. TODO: More advanced routing method can be employed.

  // Simplified Assumption Three: All routers have infinite storage that can
  // hold all the packets to be transferred;

  int x_inc = (x_from < x_to) ? 1 : -1;
  int y_inc = (y_from < y_to) ? 1 : -1;
  // Starting point:
  timeTable[{x_from, y_from}].start =
      timeTable[{x_from, y_from}].end + transfer_latency;
  timeTable[{x_from, y_from}].end =
      timeTable[{x_from, y_from}].start + send_latency;
  TileLocation nxtLoc = (y_from == y_to) ? TileLocation{x_from + x_inc, y_from}
                                         : TileLocation{x_from, y_from + y_inc};
  if (timeTable[{x_from, y_from}].start < timeTable[nxtLoc].end) {
    // Congestion happened
    timeTable[{x_from, y_from}].end = timeTable[nxtLoc].end + send_latency;
  }

  // For the rest points, use three "locations" to iteratively find the answer.
  TileLocation thisLoc = nxtLoc;
  TileLocation preLoc = TileLocation{x_from, y_from};
  nxtLoc = (thisLoc == TileLocation{x_to, y_to}) ? thisLoc
           : (thisLoc.y == y_to) ? TileLocation{thisLoc.x + x_inc, thisLoc.y}
                                 : TileLocation{thisLoc.x, thisLoc.y + y_inc};

  while (thisLoc != TileLocation{x_to, y_to}) {
    if (timeTable[preLoc].start < timeTable[thisLoc].end) {
      // Congestion happened
      timeTable[thisLoc].start = timeTable[thisLoc].end + transfer_latency;
    } else {
      // No congestion
      timeTable[thisLoc].start = timeTable[preLoc].start + transfer_latency;
    }

    if (timeTable[thisLoc].start < timeTable[nxtLoc].end) {
      // Congestion happened
      timeTable[thisLoc].end = timeTable[nxtLoc].end + send_latency;
    } else {
      // No congestion
      timeTable[thisLoc].end = timeTable[thisLoc].start + send_latency;
    }
    preLoc = thisLoc;
    thisLoc = nxtLoc;
    nxtLoc = (thisLoc == TileLocation{x_to, y_to}) ? thisLoc
             : (thisLoc.y == y_to) ? TileLocation{thisLoc.x + x_inc, thisLoc.y}
                                   : TileLocation{thisLoc.x, thisLoc.y + y_inc};
  }

  // Jump out of the loop, nxtLoc = thisLoc = TileLocation{x_to, y_to};
  if (timeTable[preLoc].start < timeTable[{x_to, y_to}].end) {
    // Congestion happened
    timeTable[{x_to, y_to}].start =
        timeTable[{x_to, y_to}].end + transfer_latency;
  } else {
    // No congestion
    timeTable[{x_to, y_to}].start = timeTable[preLoc].start + transfer_latency;
  }

  timeTable[{x_to, y_to}].end = timeTable[{x_to, y_to}].start + send_latency;

  return timeTable[{x_to, y_to}].end;
}

/**
 * @brief Simplified version of schedule_idle, in order to accelerate schedule
 * efficiency in random_pipeline.
 *
 */
double Mesh_Placing::schedule_idle_simplified(int x_to, int y_to, int x_from,
                                              int y_from, double send_latency,
                                              double transfer_latency) {
  
  return 0.0;
}

/**
 * @brief test schedule_idle() in Mesh_Placing. Given an ordered set of traffic
 * routes, calculate the total traffic time of NoC.
 *
 */
void Mesh_Placing::test_schedule_idle() {
  cout << "test_schedule_idle!!" << endl;
  double res;
  // (1,1) -> (3,2), send = 5, transfer = 1
  res = schedule_idle(3, 2, 1, 1, 5, 1);
  cout << "Schedule 1, routing latency = " << res << endl;
  // (1,1) -> (1,3), send = 4, transfer = 1
  res = schedule_idle(1, 3, 1, 1, 4, 1);
  cout << "Schedule 2, routing latency = " << res << endl;
  // (2,1) -> (0,2), send = 5, transfer = 1
  res = schedule_idle(0, 2, 2, 1, 5, 1);
  cout << "Schedule 3, routing latency = " << res << endl;
  // (3,1) -> (3,3), send = 4, transfer = 1
  res = schedule_idle(3, 3, 3, 1, 4, 1);
  cout << "Schedule 4, routing latency = " << res << endl;
  // (2,0) -> (0,0), send = 8, transfer = 1
  res = schedule_idle(0, 0, 2, 0, 8, 1);
  cout << "Schedule 5, routing latency = " << res << endl;
  cout << "********** timeTable **********" << endl;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      cout << "timeTable[" << i << "][" << j
           << "] = " << timeTable[{i, j}].start << "," << timeTable[{i, j}].end
           << endl;
    }
  }
}

void Mesh_Placing::test_NMAP() {
  ifstream f_traffic(PATH_TRAFFIC);
  int src, dst, packet;
  map<pair<int, int>, int> traffic_mp; // The data transfer between x and y.
  while (f_traffic >> src >> dst >> packet) {
    traffic_mp[{src, dst}] = packet;
  }

  ifstream f_tile(PATH_TILENUM);
  f_tile >> totalTiles;
  vector<TileLocation> tileLocation;
  tileLocation.resize(totalTiles);
  NMAP(traffic_mp, tileLocation);
  ofstream f_place(PATH_PLACING);
  for (int i = 0; i < totalTiles; i++) {
    f_place << "tile: " << i << " location: [" << tileLocation[i].x << ", "
            << tileLocation[i].y << "]" << endl;
  }

  f_traffic.clear();
  f_traffic.seekg(0, std::ios::beg);
  vector<vector<int>> traffic;
  while (f_traffic >> src >> dst >> packet) {
    traffic.push_back({src, dst, packet});
  }

  NMAP_pipeline(traffic, tileLocation);

  // int tile_rows = floor(sqrt(totalTiles));
  // int tile_cols = ceil((double)totalTiles / tile_rows);
  // int x = 0, y = 0;
  // for (int i = 0; i < totalTiles; i++) {
  //   tileLocation[i] = {x, y};
  //   y++;
  //   if (y >= tile_cols) {
  //     y = 0;
  //     x++;
  //   }
  // }

  // random_pipeline(traffic, tileLocation);
}

} // namespace Refactor
