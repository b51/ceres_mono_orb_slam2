/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: LoopClosing.h
 *
 *          Created On: Wed 04 Sep 2019 05:35:12 PM CST
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

#ifndef LOOPCLOSING_H_
#define LOOPCLOSING_H_

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"

#include "KeyFrameDatabase.h"
// #include "CeresSim3.h"
#include <sophus/sim3.hpp>

#include <mutex>
#include <thread>

namespace ORB_SLAM2 {

class Tracking;
class LocalMapping;
class KeyFrameDatabase;

class LoopClosing {
 public:
  typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;

  typedef std::map<
      KeyFrame*, Sophus::Sim3d, std::less<KeyFrame*>,
      Eigen::aligned_allocator<std::pair<const KeyFrame*, Sophus::Sim3d> > >
      KeyFrameAndSim3;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  LoopClosing(Map* map, KeyFrameDatabase* keyframe_database,
              ORBVocabulary* orb_vocabulary, const bool is_fix_scale);

  void SetTracker(Tracking* tracker);

  void SetLocalMapper(LocalMapping* local_mapper);

  // Main function
  void Run();

  void InsertKeyFrame(KeyFrame* keyframe);

  void RequestReset();

  // This function will run in a separate thread
  void RunGlobalBundleAdjustment(unsigned long nLoopKF);

  bool isRunningGBA() {
    unique_lock<std::mutex> lock(mutex_global_BA_);
    return is_running_global_BA_;
  }
  bool isFinishedGBA() {
    unique_lock<std::mutex> lock(mutex_global_BA_);
    return is_finished_global_BA_;
  }

  void RequestFinish();

  bool isFinished();

 protected:
  bool CheckNewKeyFrames();

  bool DetectLoop();

  bool ComputeSim3();

  void SearchAndFuse(const KeyFrameAndSim3& CorrectedPosesMap);

  void CorrectLoop();

  void ResetIfRequested();
  bool is_reset_requested_;
  std::mutex mutex_reset_;

  bool CheckFinish();
  void SetFinish();
  bool is_finish_requested_;
  bool is_finished_;
  std::mutex mutex_finish_;

  Map* map_;
  Tracking* tracker_;

  KeyFrameDatabase* keyframe_database_;
  ORBVocabulary* orb_vocabulary_;

  LocalMapping* local_mapper_;

  std::list<KeyFrame*> loop_keyframe_queue_;

  std::mutex mutex_loop_queue_;

  // Loop detector parameters
  float covisibility_consistency_threshold_;

  // Loop detector variables
  KeyFrame* current_keyframe_;
  KeyFrame* matched_keyframe_;
  std::vector<ConsistentGroup> consistent_groups_;
  std::vector<KeyFrame*> enough_consistent_candidates_;
  std::vector<KeyFrame*> current_connected_keyframes_;
  std::vector<MapPoint*> current_matched_map_points_;
  std::vector<MapPoint*> loop_map_points_;
  Eigen::Matrix4d Scw_;
  Sophus::Sim3d sophus_sim3_Scw_;

  long unsigned int last_loop_keyframe_id_;

  // Variables related to Global Bundle Adjustment
  bool is_running_global_BA_;
  bool is_finished_global_BA_;
  bool is_stop_global_BA_;
  std::mutex mutex_global_BA_;
  std::thread* thread_global_BA_;

  // Fix scale in the stereo/RGB-D case
  bool is_fix_scale_;

  int full_BA_index_;
};
}  // namespace ORB_SLAM2

#endif  // LOOPCLOSING_H
