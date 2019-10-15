/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: MapPoint.cc
 *
 *          Created On: Thu 05 Sep 2019 10:37:43 AM CST
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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include <mutex>

namespace ORB_SLAM2 {

long unsigned int MapPoint::next_id_ = 0;
mutex MapPoint::global_mutex_;

MapPoint::MapPoint(const Eigen::Vector3d& pos, KeyFrame* reference_keyframe,
                   Map* map)
    : first_keyframe_id_(reference_keyframe->id_),
      first_frame_id_(reference_keyframe->frame_id_),
      n_observations_(0),
      track_reference_for_frame_(0),
      last_seen_frame_id_(0),
      n_BA_local_for_keyframe_(0),
      fuse_candidate_for_keyframe(0),
      loop_point_for_keyframe_(0),
      corrected_by_keyframe_(0),
      corrected_reference_(0),
      n_BA_global_for_keyframe_(0),
      reference_keyframe_(reference_keyframe),
      n_visible_(1),
      n_found_(1),
      is_bad_(false),
      replaced_map_point_(static_cast<MapPoint*>(nullptr)),
      min_distance_(0),
      max_distance_(0),
      map_(map) {
  world_pose_ = pos;
  normal_vector_ = Eigen::Vector3d::Zero();

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid
  // conflicts with id.
  unique_lock<mutex> lock(map_->mutex_point_creation_);
  id_ = next_id_++;
}

MapPoint::MapPoint(const Eigen::Vector3d& pos, Map* map, Frame* frame,
                   const int& idxF)
    : first_keyframe_id_(-1),
      first_frame_id_(frame->id_),
      n_observations_(0),
      track_reference_for_frame_(0),
      last_seen_frame_id_(0),
      n_BA_local_for_keyframe_(0),
      fuse_candidate_for_keyframe(0),
      loop_point_for_keyframe_(0),
      corrected_by_keyframe_(0),
      corrected_reference_(0),
      n_BA_global_for_keyframe_(0),
      reference_keyframe_(static_cast<KeyFrame*>(nullptr)),
      n_visible_(1),
      n_found_(1),
      is_bad_(false),
      replaced_map_point_(nullptr),
      map_(map) {
  world_pose_ = pos;
  Eigen::Vector3d Ow = frame->GetCameraCenter();
  normal_vector_ = world_pose_ - Ow;
  normal_vector_ = normal_vector_ / normal_vector_.norm();

  Eigen::Vector3d PC = pos - Ow;
  const float dist = PC.norm();
  const int level = frame->undistort_keypoints_[idxF].octave;
  const float level_scale_factor = frame->scale_factors_[level];
  const int nLevels = frame->n_scale_levels_;

  max_distance_ = dist * level_scale_factor;
  min_distance_ = max_distance_ / frame->scale_factors_[nLevels - 1];

  frame->descriptors_.row(idxF).copyTo(descriptor_);

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid
  // conflicts with id.
  unique_lock<mutex> lock(map_->mutex_point_creation_);
  id_ = next_id_++;
}

void MapPoint::SetWorldPos(const Eigen::Vector3d& pos) {
  unique_lock<mutex> lock2(global_mutex_);
  unique_lock<mutex> lock(mutex_pose_);
  world_pose_ = pos;
}

Eigen::Vector3d MapPoint::GetWorldPos() {
  unique_lock<mutex> lock(mutex_pose_);
  return world_pose_;
}

Eigen::Vector3d MapPoint::GetNormal() {
  unique_lock<mutex> lock(mutex_pose_);
  return normal_vector_;
}

KeyFrame* MapPoint::GetReferenceKeyFrame() {
  unique_lock<mutex> lock(mutex_features_);
  return reference_keyframe_;
}

void MapPoint::AddObservation(KeyFrame* keyframe, size_t index) {
  unique_lock<mutex> lock(mutex_features_);
  if (observations_.count(keyframe)) return;
  observations_[keyframe] = index;
  n_observations_++;
}

void MapPoint::EraseObservation(KeyFrame* keyframe) {
  bool is_bad = false;
  {
    unique_lock<mutex> lock(mutex_features_);
    if (observations_.count(keyframe)) {
      int index = observations_[keyframe];
      if (keyframe->mvuRight[index] >= 0)
        n_observations_ -= 2;
      else
        n_observations_--;

      observations_.erase(keyframe);

      if (reference_keyframe_ == keyframe)
        reference_keyframe_ = observations_.begin()->first;

      // If only 2 observations or less, discard point
      if (n_observations_ <= 2) is_bad = true;
    }
  }

  if (is_bad) SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations() {
  unique_lock<mutex> lock(mutex_features_);
  return observations_;
}

int MapPoint::Observations() {
  unique_lock<mutex> lock(mutex_features_);
  return n_observations_;
}

void MapPoint::SetBadFlag() {
  map<KeyFrame*, size_t> observations;
  {
    unique_lock<mutex> lock1(mutex_features_);
    unique_lock<mutex> lock2(mutex_pose_);
    is_bad_ = true;
    observations = observations_;
    observations_.clear();
  }
  for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                        mend = observations.end();
       mit != mend; mit++) {
    KeyFrame* keyframe = mit->first;
    keyframe->EraseMapPointMatch(mit->second);
  }

  map_->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced() {
  unique_lock<mutex> lock1(mutex_features_);
  unique_lock<mutex> lock2(mutex_pose_);
  return replaced_map_point_;
}

void MapPoint::Replace(MapPoint* map_point) {
  if (map_point->id_ == this->id_) return;

  int nvisible, nfound;
  map<KeyFrame*, size_t> observations;
  {
    unique_lock<mutex> lock1(mutex_features_);
    unique_lock<mutex> lock2(mutex_pose_);
    observations = observations_;
    observations_.clear();
    is_bad_ = true;
    nvisible = n_visible_;
    nfound = n_found_;
    replaced_map_point_ = map_point;
  }

  for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                        mend = observations.end();
       mit != mend; mit++) {
    // Replace measurement in keyframe
    KeyFrame* keyframe = mit->first;

    if (!map_point->IsInKeyFrame(keyframe)) {
      keyframe->ReplaceMapPointMatch(mit->second, map_point);
      map_point->AddObservation(keyframe, mit->second);
    } else {
      keyframe->EraseMapPointMatch(mit->second);
    }
  }
  map_point->IncreaseFound(nfound);
  map_point->IncreaseVisible(nvisible);
  map_point->ComputeDistinctiveDescriptors();

  map_->EraseMapPoint(this);
}

bool MapPoint::isBad() {
  unique_lock<mutex> lock(mutex_features_);
  unique_lock<mutex> lock2(mutex_pose_);
  return is_bad_;
}

void MapPoint::IncreaseVisible(int n) {
  unique_lock<mutex> lock(mutex_features_);
  n_visible_ += n;
}

void MapPoint::IncreaseFound(int n) {
  unique_lock<mutex> lock(mutex_features_);
  n_found_ += n;
}

float MapPoint::GetFoundRatio() {
  unique_lock<mutex> lock(mutex_features_);
  return static_cast<float>(n_found_) / n_visible_;
}

void MapPoint::ComputeDistinctiveDescriptors() {
  // Retrieve all observed descriptors
  vector<cv::Mat> descriptors;

  map<KeyFrame*, size_t> observations;

  {
    unique_lock<mutex> lock1(mutex_features_);
    if (is_bad_) return;
    observations = observations_;
  }

  if (observations.empty()) return;

  descriptors.reserve(observations.size());

  for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                        mend = observations.end();
       mit != mend; mit++) {
    KeyFrame* keyframe = mit->first;

    if (!keyframe->isBad())
      descriptors.push_back(keyframe->descriptors_.row(mit->second));
  }

  if (descriptors.empty()) return;

  // Compute distances between them
  const size_t N = descriptors.size();

  float distances[N][N];
  for (size_t i = 0; i < N; i++) {
    distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
      int distij =
          ORBmatcher::DescriptorDistance(descriptors[i], descriptors[j]);
      distances[i][j] = distij;
      distances[j][i] = distij;
    }
  }

  // Take the descriptor with least median distance to the rest
  int best_median = INT_MAX;
  int best_index = 0;
  for (size_t i = 0; i < N; i++) {
    vector<int> dists(distances[i], distances[i] + N);
    sort(dists.begin(), dists.end());
    int median = dists[0.5 * (N - 1)];

    if (median < best_median) {
      best_median = median;
      best_index = i;
    }
  }

  {
    unique_lock<mutex> lock(mutex_features_);
    descriptor_ = descriptors[best_index].clone();
  }
}

cv::Mat MapPoint::GetDescriptor() {
  unique_lock<mutex> lock(mutex_features_);
  return descriptor_.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_features_);
  if (observations_.count(keyframe))
    return observations_[keyframe];
  else
    return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_features_);
  return (observations_.count(keyframe));
}

void MapPoint::UpdateNormalAndDepth() {
  map<KeyFrame*, size_t> observations;
  KeyFrame* reference_keyframe;
  Eigen::Vector3d pos;
  {
    unique_lock<mutex> lock1(mutex_features_);
    unique_lock<mutex> lock2(mutex_pose_);
    if (is_bad_) return;
    observations = observations_;
    reference_keyframe = reference_keyframe_;
    pos = world_pose_;
  }

  if (observations.empty()) return;

  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  int n = 0;
  for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                        mend = observations.end();
       mit != mend; mit++) {
    KeyFrame* keyframe = mit->first;
    Eigen::Vector3d Owi = keyframe->GetCameraCenter();
    Eigen::Vector3d normali = world_pose_ - Owi;
    normal = normal + normali / normali.norm();
    n++;
  }

  Eigen::Vector3d Ow = reference_keyframe->GetCameraCenter();
  Eigen::Vector3d PC = pos - Ow;
  const float dist = PC.norm();
  const int level =
      reference_keyframe->undistort_keypoints_[observations[reference_keyframe]]
          .octave;
  const float level_scale_factor = reference_keyframe->scale_factors_[level];
  const int nLevels = reference_keyframe->n_scale_levels_;

  {
    unique_lock<mutex> lock3(mutex_pose_);
    max_distance_ = dist * level_scale_factor;
    min_distance_ =
        max_distance_ / reference_keyframe->scale_factors_[nLevels - 1];
    normal_vector_ = normal / n;
  }
}

float MapPoint::GetMinDistanceInvariance() {
  unique_lock<mutex> lock(mutex_pose_);
  return 0.8f * min_distance_;
}

float MapPoint::GetMaxDistanceInvariance() {
  unique_lock<mutex> lock(mutex_pose_);
  return 1.2f * max_distance_;
}

int MapPoint::PredictScale(const float& current_dist, KeyFrame* keyframe) {
  float ratio;
  {
    unique_lock<mutex> lock(mutex_pose_);
    ratio = max_distance_ / current_dist;
  }

  int nScale = ceil(log(ratio) / keyframe->log_scale_factor_);
  if (nScale < 0)
    nScale = 0;
  else if (nScale >= keyframe->n_scale_levels_)
    nScale = keyframe->n_scale_levels_ - 1;

  return nScale;
}

int MapPoint::PredictScale(const float& current_dist, Frame* frame) {
  float ratio;
  {
    unique_lock<mutex> lock(mutex_pose_);
    ratio = max_distance_ / current_dist;
  }

  int nScale = ceil(log(ratio) / frame->log_scale_factor_);
  if (nScale < 0)
    nScale = 0;
  else if (nScale >= frame->n_scale_levels_)
    nScale = frame->n_scale_levels_ - 1;

  return nScale;
}

}  // namespace ORB_SLAM2
