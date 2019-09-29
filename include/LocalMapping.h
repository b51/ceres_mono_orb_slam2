/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: LocalMapping.h
 *
 *          Created On: Wed 04 Sep 2019 04:37:00 PM CST
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

#ifndef LOCALMAPPING_H_
#define LOCALMAPPING_H_

#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "LoopClosing.h"
#include "Map.h"
#include "Tracking.h"

#include <Eigen/Geometry>
#include <mutex>

namespace ORB_SLAM2 {

class Tracking;
class LoopClosing;
class Map;

class LocalMapping {
 public:
  LocalMapping(Map* map, const float is_monocular);

  void SetLoopCloser(LoopClosing* loop_closer);

  void SetTracker(Tracking* tracker);

  // Main function
  void Run();

  void InsertKeyFrame(KeyFrame* keyframe);

  // Thread Synch
  void RequestStop();
  void RequestReset();
  bool Stop();
  void Release();
  bool isStopped();
  bool stopRequested();
  bool AcceptKeyFrames();
  void SetAcceptKeyFrames(bool flag);
  bool SetNotStop(bool flag);

  void InterruptBA();

  void RequestFinish();
  bool isFinished();

  int KeyframesInQueue() {
    unique_lock<mutex> lock(mutex_new_keyframes_);
    return new_keyframes_.size();
  }

 protected:
  bool CheckNewKeyFrames();
  void ProcessNewKeyFrame();
  void CreateNewMapPoints();

  void MapPointCulling();
  void SearchInNeighbors();

  void KeyFrameCulling();

  Eigen::Matrix3d ComputeF12(KeyFrame*& pKF1, KeyFrame*& pKF2);

  Eigen::Matrix3d SkewSymmetricMatrix(const Eigen::Vector3d& v);

  bool is_monocular_;

  void ResetIfRequested();
  bool is_reset_requested_;
  std::mutex mutex_reset_;

  bool CheckFinish();
  void SetFinish();
  bool is_finish_requested_;
  bool is_finished_;
  std::mutex mutex_finish_;

  Map* map_;

  LoopClosing* loop_closer_;
  Tracking* tracker_;

  std::list<KeyFrame*> new_keyframes_;

  KeyFrame* current_keyframe_;

  std::list<MapPoint*> recent_added_map_points_;

  std::mutex mutex_new_keyframes_;

  bool is_abort_BA_;

  bool is_stopped_;
  bool is_stop_requested_;
  bool do_not_stop_;
  std::mutex mutex_stop_;

  bool do_accept_keyframes_;
  std::mutex mutex_accept_;
};
}  // namespace ORB_SLAM2

#endif
