/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: MapPoint.h
 *
 *          Created On: Thu 05 Sep 2019 10:38:24 AM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/
/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MAPPOINT_H_
#define MAPPOINT_H_

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"

#include <mutex>
#include <map>
#include <opencv2/core/core.hpp>
#include <Eigen/Geometry>

namespace ORB_SLAM2 {

class KeyFrame;
class Map;
class Frame;

class MapPoint {
 public:
  MapPoint(const Eigen::Vector3d& pos, KeyFrame* reference_keyframe, Map* map);
  MapPoint(const Eigen::Vector3d& pos, Map* map, Frame* frame, const int& idxF);

  void SetWorldPos(const Eigen::Vector3d& pos);
  Eigen::Vector3d GetWorldPos();

  Eigen::Vector3d GetNormal();
  KeyFrame* GetReferenceKeyFrame();

  std::map<KeyFrame*, size_t> GetObservations();
  int Observations();

  void AddObservation(KeyFrame* keyframe, size_t index);
  void EraseObservation(KeyFrame* keyframe);

  int GetIndexInKeyFrame(KeyFrame* keyframe);
  bool IsInKeyFrame(KeyFrame* keyframe);

  void SetBadFlag();
  bool isBad();

  void Replace(MapPoint* pMP);
  MapPoint* GetReplaced();

  void IncreaseVisible(int n = 1);
  void IncreaseFound(int n = 1);
  float GetFoundRatio();
  inline int GetFound() { return n_found_; }

  void ComputeDistinctiveDescriptors();

  cv::Mat GetDescriptor();

  void UpdateNormalAndDepth();

  float GetMinDistanceInvariance();
  float GetMaxDistanceInvariance();
  int PredictScale(const float& currentDist, KeyFrame* keyframe);
  int PredictScale(const float& currentDist, Frame* pF);

  static bool lId(MapPoint* map_point_1, MapPoint* map_point_2) {
    return map_point_1->id_ < map_point_2->id_;
  }

 public:
  long unsigned int id_;
  static long unsigned int next_id_;
  long int first_keyframe_id_;
  long int first_frame_id_;
  int n_observations_;

  // Variables used by the tracking
  float track_proj_x_;
  float track_proj_y_;
  float track_proj_x_r_;
  bool is_track_in_view_;
  int track_scale_level_;
  float track_view_cos_;
  long unsigned int track_reference_for_frame_;
  long unsigned int last_seen_frame_id_;

  // Variables used by local mapping
  long unsigned int n_BA_local_for_keyframe_;
  long unsigned int fuse_candidate_for_keyframe;

  // Variables used by loop closing
  long unsigned int loop_point_for_keyframe_;
  long unsigned int corrected_by_keyframe_;
  long unsigned int corrected_reference_;
  Eigen::Vector3d global_BA_pose_;
  long unsigned n_BA_global_for_keyframe_;

  static std::mutex global_mutex_;

 protected:
  // Position in absolute coordinates
  Eigen::Vector3d world_pose_;

  // Keyframes observing the point and associated index in keyframe
  std::map<KeyFrame*, size_t> observations_;

  // Mean viewing direction
  Eigen::Vector3d normal_vector_;

  // Best descriptor to fast matching
  cv::Mat descriptor_;

  // Reference KeyFrame
  KeyFrame* reference_keyframe_;

  // Tracking counters
  int n_visible_;
  int n_found_;

  // Bad flag (we do not currently erase MapPoint from memory)
  bool is_bad_;
  MapPoint* replaced_map_point_;

  // Scale invariance distances
  float min_distance_;
  float max_distance_;

  Map* map_;

  std::mutex mutex_pose_;
  std::mutex mutex_features_;
};

}  // namespace ORB_SLAM2

#endif  // MAPPOINT_H
