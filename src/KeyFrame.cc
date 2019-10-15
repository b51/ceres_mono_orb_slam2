/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: KeyFrame.cc
 *
 *          Created On: Wed 04 Sep 2019 12:02:32 PM CST
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

#include "KeyFrame.h"
#include <mutex>
#include "MatEigenConverter.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {

long unsigned int KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(Frame& frame, Map* map, KeyFrameDatabase* keyframe_database)
    : frame_id_(frame.id_),
      timestamp_(frame.timestamp_),
      grid_cols_(FRAME_GRID_COLS),
      grid_rows_(FRAME_GRID_ROWS),
      grid_element_width_inv_(frame.grid_element_width_inv_),
      grid_element_height_inv_(frame.grid_element_height_inv_),
      track_reference_for_frame_(0),
      fuse_target_for_keyframe_(0),
      n_BA_local_for_keyframe_(0),
      n_BA_fixed_for_keyframe_(0),
      n_loop_query_(0),
      n_loop_words_(0),
      reloc_query_(0),
      n_reloc_words_(0),
      n_BA_global_for_keyframe_(0),
      fx_(frame.fx_),
      fy_(frame.fy_),
      cx_(frame.cx_),
      cy_(frame.cy_),
      invfx_(frame.invfx_),
      invfy_(frame.invfy_),
      bf_(frame.bf_),
      mb_(frame.mb_),
      threshold_depth_(frame.threshold_depth_),
      N_(frame.N_),
      keypoints_(frame.keypoints_),
      undistort_keypoints_(frame.undistort_keypoints_),
      mvuRight(frame.mvuRight),
      depthes_(frame.depthes_),
      descriptors_(frame.descriptors_.clone()),
      bow_vector_(frame.bow_vector_),
      feature_vector_(frame.feature_vector_),
      n_scale_levels_(frame.n_scale_levels_),
      scale_factor_(frame.scale_factor_),
      log_scale_factor_(frame.log_scale_factor_),
      scale_factors_(frame.scale_factors_),
      level_sigma2s_(frame.level_sigma2s_),
      inv_level_sigma2s_(frame.inv_level_sigma2s_),
      min_x_(frame.min_x_),
      min_y_(frame.min_y_),
      max_x_(frame.max_x_),
      max_y_(frame.max_y_),
      K_(frame.K_),
      map_points_(frame.map_points_),
      keyframe_database_(keyframe_database),
      orb_vocabulary_(frame.orb_vocabulary_),
      is_first_connection_(true),
      parent_(nullptr),
      do_not_erase_(false),
      do_to_be_erased_(false),
      is_bad_(false),
      half_baseline_(frame.mb_ / 2),
      map_(map) {
  id_ = next_id_++;

  grid_.resize(grid_cols_);
  for (int i = 0; i < grid_cols_; i++) {
    grid_[i].resize(grid_rows_);
    for (int j = 0; j < grid_rows_; j++) {
      grid_[i][j] = frame.grid_[i][j];
    }
  }

  SetPose(frame.Tcw_);
}

void KeyFrame::ComputeBoW() {
  if (bow_vector_.empty() || feature_vector_.empty()) {
    vector<cv::Mat> current_descriptors =
        MatEigenConverter::toDescriptorVector(descriptors_);
    // Feature vector associate features with nodes in the 4th level (from
    // leaves up) We assume the vocabulary tree has 6 levels, change the 4
    // otherwise
    orb_vocabulary_->transform(current_descriptors, bow_vector_,
                               feature_vector_, 4);
  }
}

void KeyFrame::SetPose(const Eigen::Matrix4d& Tcw) {
  unique_lock<mutex> lock(mutex_pose_);
  Tcw_ = Tcw;
  Eigen::Matrix3d Rcw = Tcw_.block<3, 3>(0, 0);
  Eigen::Vector3d tcw = Tcw_.block<3, 1>(0, 3);
  Eigen::Matrix3d Rwc = Rcw.transpose();
  Ow_ = -Rwc * tcw;

  Twc_ = Eigen::Matrix4d::Identity();
  Twc_.block<3, 3>(0, 0) = Rwc;
  Twc_.block<3, 1>(0, 3) = Ow_;
  Eigen::Vector4d center;
}

Eigen::Matrix4d KeyFrame::GetPose() {
  unique_lock<mutex> lock(mutex_pose_);
  return Tcw_;
}

Eigen::Matrix4d KeyFrame::GetPoseInverse() {
  unique_lock<mutex> lock(mutex_pose_);
  return Twc_;
}

Eigen::Vector3d KeyFrame::GetCameraCenter() {
  unique_lock<mutex> lock(mutex_pose_);
  return Ow_;
}

Eigen::Matrix3d KeyFrame::GetRotation() {
  unique_lock<mutex> lock(mutex_pose_);
  return Tcw_.block<3, 3>(0, 0);
}

Eigen::Vector3d KeyFrame::GetTranslation() {
  unique_lock<mutex> lock(mutex_pose_);
  return Tcw_.block<3, 1>(0, 3);
}

void KeyFrame::AddConnection(KeyFrame* keyframe, const int& weight) {
  {
    unique_lock<mutex> lock(mutex_connections_);
    if (!connected_keyframe_weights_.count(keyframe))
      connected_keyframe_weights_[keyframe] = weight;
    else if (connected_keyframe_weights_[keyframe] != weight)
      connected_keyframe_weights_[keyframe] = weight;
    else
      return;
  }

  UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles() {
  unique_lock<mutex> lock(mutex_connections_);
  vector<pair<int, KeyFrame*> > pairs;
  pairs.reserve(connected_keyframe_weights_.size());
  for (std::map<KeyFrame*, int>::iterator
           mit = connected_keyframe_weights_.begin(),
           mend = connected_keyframe_weights_.end();
       mit != mend; mit++)
    pairs.push_back(make_pair(mit->second, mit->first));

  sort(pairs.begin(), pairs.end());
  list<KeyFrame*> keyframe_list;
  list<int> weight_list;
  for (size_t i = 0, iend = pairs.size(); i < iend; i++) {
    keyframe_list.push_front(pairs[i].second);
    weight_list.push_front(pairs[i].first);
  }

  ordered_connected_keyframes_ =
      vector<KeyFrame*>(keyframe_list.begin(), keyframe_list.end());
  ordered_weights_ = vector<int>(weight_list.begin(), weight_list.end());
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames() {
  unique_lock<mutex> lock(mutex_connections_);
  set<KeyFrame*> keyframe_set;
  for (map<KeyFrame*, int>::iterator mit = connected_keyframe_weights_.begin();
       mit != connected_keyframe_weights_.end(); mit++)
    keyframe_set.insert(mit->first);
  return keyframe_set;
}

std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames() {
  unique_lock<mutex> lock(mutex_connections_);
  return ordered_connected_keyframes_;
}

std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int& n) {
  unique_lock<mutex> lock(mutex_connections_);
  if ((int)ordered_connected_keyframes_.size() < n)
    return ordered_connected_keyframes_;
  else
    return std::vector<KeyFrame*>(ordered_connected_keyframes_.begin(),
                                  ordered_connected_keyframes_.begin() + n);
}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int& w) {
  unique_lock<mutex> lock(mutex_connections_);

  if (ordered_connected_keyframes_.empty()) {
    return std::vector<KeyFrame*>();
  }

  vector<int>::iterator it =
      upper_bound(ordered_weights_.begin(), ordered_weights_.end(), w,
                  KeyFrame::weightComp);
  if (it == ordered_weights_.end())
    return std::vector<KeyFrame*>();
  else {
    int n = it - ordered_weights_.begin();
    return std::vector<KeyFrame*>(ordered_connected_keyframes_.begin(),
                                  ordered_connected_keyframes_.begin() + n);
  }
}

int KeyFrame::GetWeight(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_connections_);
  if (connected_keyframe_weights_.count(keyframe))
    return connected_keyframe_weights_[keyframe];
  else
    return 0;
}

void KeyFrame::AddMapPoint(MapPoint* map_point, const size_t& index) {
  unique_lock<mutex> lock(mutex_features_);
  map_points_[index] = map_point;
}

void KeyFrame::EraseMapPointMatch(const size_t& index) {
  unique_lock<mutex> lock(mutex_features_);
  map_points_[index] = static_cast<MapPoint*>(nullptr);
}

void KeyFrame::EraseMapPointMatch(MapPoint* map_point) {
  int index = map_point->GetIndexInKeyFrame(this);
  if (index >= 0) {
    map_points_[index] = static_cast<MapPoint*>(nullptr);
  }
}

void KeyFrame::ReplaceMapPointMatch(const size_t& index, MapPoint* map_point) {
  map_points_[index] = map_point;
}

std::set<MapPoint*> KeyFrame::GetMapPoints() {
  unique_lock<mutex> lock(mutex_features_);
  std::set<MapPoint*> map_point_set;
  for (size_t i = 0, iend = map_points_.size(); i < iend; i++) {
    if (!map_points_[i]) {
      continue;
    }
    MapPoint* map_point = map_points_[i];
    if (!map_point->isBad()) {
      map_point_set.insert(map_point);
    }
  }
  return map_point_set;
}

int KeyFrame::TrackedMapPoints(const int& min_obs) {
  unique_lock<mutex> lock(mutex_features_);

  int nPoints = 0;
  const bool do_check_obs = min_obs > 0;
  for (int i = 0; i < N_; i++) {
    MapPoint* map_point = map_points_[i];
    if (map_point) {
      if (!map_point->isBad()) {
        if (do_check_obs) {
          if (map_points_[i]->Observations() >= min_obs) {
            nPoints++;
          }
        } else {
          nPoints++;
        }
      }
    }
  }

  return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches() {
  unique_lock<mutex> lock(mutex_features_);
  return map_points_;
}

MapPoint* KeyFrame::GetMapPoint(const size_t& index) {
  unique_lock<mutex> lock(mutex_features_);
  return map_points_[index];
}

void KeyFrame::UpdateConnections() {
  std::map<KeyFrame*, int> keyframe_counter;

  vector<MapPoint*> map_points;

  {
    unique_lock<mutex> lockMPs(mutex_features_);
    map_points = map_points_;
  }

  // For all map points in keyframe check in which other keyframes are they seen
  // Increase counter for those keyframes
  for (vector<MapPoint*>::iterator vit = map_points.begin(),
                                   vend = map_points.end();
       vit != vend; vit++) {
    MapPoint* map_point = *vit;

    if (!map_point) continue;

    if (map_point->isBad()) continue;

    std::map<KeyFrame*, size_t> observations = map_point->GetObservations();

    for (map<KeyFrame*, size_t>::iterator mit = observations.begin(),
                                          mend = observations.end();
         mit != mend; mit++) {
      if (mit->first->id_ == id_) continue;
      keyframe_counter[mit->first]++;
    }
  }

  // This should not happen
  if (keyframe_counter.empty()) return;

  // If the counter is greater than threshold add connection
  // In case no keyframe counter is over threshold add the one with maximum
  // counter
  int nmax = 0;
  KeyFrame* keyframe_max = nullptr;
  int th = 15;

  vector<pair<int, KeyFrame*> > pairs;
  pairs.reserve(keyframe_counter.size());
  for (map<KeyFrame*, int>::iterator mit = keyframe_counter.begin(),
                                     mend = keyframe_counter.end();
       mit != mend; mit++) {
    if (mit->second > nmax) {
      nmax = mit->second;
      keyframe_max = mit->first;
    }
    if (mit->second >= th) {
      pairs.push_back(make_pair(mit->second, mit->first));
      (mit->first)->AddConnection(this, mit->second);
    }
  }

  if (pairs.empty()) {
    pairs.push_back(make_pair(nmax, keyframe_max));
    keyframe_max->AddConnection(this, nmax);
  }

  sort(pairs.begin(), pairs.end());
  list<KeyFrame*> keyframe_list;
  list<int> weight_list;
  for (size_t i = 0; i < pairs.size(); i++) {
    keyframe_list.push_front(pairs[i].second);
    weight_list.push_front(pairs[i].first);
  }

  {
    unique_lock<mutex> lockCon(mutex_connections_);

    // mspConnectedKeyFrames = spConnectedKeyFrames;
    connected_keyframe_weights_ = keyframe_counter;
    ordered_connected_keyframes_ =
        vector<KeyFrame*>(keyframe_list.begin(), keyframe_list.end());
    ordered_weights_ = vector<int>(weight_list.begin(), weight_list.end());

    if (is_first_connection_ && id_ != 0) {
      parent_ = ordered_connected_keyframes_.front();
      parent_->AddChild(this);
      is_first_connection_ = false;
    }
  }
}

void KeyFrame::AddChild(KeyFrame* keyframe) {
  unique_lock<mutex> lockCon(mutex_connections_);
  childrens_.insert(keyframe);
}

void KeyFrame::EraseChild(KeyFrame* keyframe) {
  unique_lock<mutex> lockCon(mutex_connections_);
  childrens_.erase(keyframe);
}

void KeyFrame::ChangeParent(KeyFrame* keyframe) {
  unique_lock<mutex> lockCon(mutex_connections_);
  parent_ = keyframe;
  keyframe->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds() {
  unique_lock<mutex> lockCon(mutex_connections_);
  return childrens_;
}

KeyFrame* KeyFrame::GetParent() {
  unique_lock<mutex> lockCon(mutex_connections_);
  return parent_;
}

bool KeyFrame::hasChild(KeyFrame* keyframe) {
  unique_lock<mutex> lockCon(mutex_connections_);
  return childrens_.count(keyframe);
}

void KeyFrame::AddLoopEdge(KeyFrame* keyframe) {
  unique_lock<mutex> lockCon(mutex_connections_);
  do_not_erase_ = true;
  loop_edges_.insert(keyframe);
}

set<KeyFrame*> KeyFrame::GetLoopEdges() {
  unique_lock<mutex> lockCon(mutex_connections_);
  return loop_edges_;
}

void KeyFrame::SetNotErase() {
  unique_lock<mutex> lock(mutex_connections_);
  do_not_erase_ = true;
}

void KeyFrame::SetErase() {
  {
    unique_lock<mutex> lock(mutex_connections_);
    if (loop_edges_.empty()) {
      do_not_erase_ = false;
    }
  }

  if (do_to_be_erased_) {
    SetBadFlag();
  }
}

void KeyFrame::SetBadFlag() {
  {
    unique_lock<mutex> lock(mutex_connections_);
    if (id_ == 0)
      return;
    else if (do_not_erase_) {
      do_to_be_erased_ = true;
      return;
    }
  }

  for (std::map<KeyFrame*, int>::iterator
           mit = connected_keyframe_weights_.begin(),
           mend = connected_keyframe_weights_.end();
       mit != mend; mit++) {
    mit->first->EraseConnection(this);
  }

  for (size_t i = 0; i < map_points_.size(); i++) {
    if (map_points_[i]) map_points_[i]->EraseObservation(this);
  }

  {
    unique_lock<mutex> lock(mutex_connections_);
    unique_lock<mutex> lock1(mutex_features_);

    connected_keyframe_weights_.clear();
    ordered_connected_keyframes_.clear();

    // Update Spanning Tree
    set<KeyFrame*> parent_candidates;
    parent_candidates.insert(parent_);

    // Assign at each iteration one children with a parent (the pair with
    // highest covisibility weight) Include that children as new parent
    // candidate for the rest
    while (!childrens_.empty()) {
      bool do_continue = false;

      int max = -1;
      KeyFrame* child;
      KeyFrame* parent;

      for (set<KeyFrame*>::iterator sit = childrens_.begin(),
                                    send = childrens_.end();
           sit != send; sit++) {
        KeyFrame* keyframe = *sit;
        if (keyframe->isBad()) continue;

        // Check if a parent candidate is connected to the keyframe
        vector<KeyFrame*> connected_keyframes =
            keyframe->GetVectorCovisibleKeyFrames();
        for (size_t i = 0, iend = connected_keyframes.size(); i < iend; i++) {
          for (std::set<KeyFrame*>::iterator spcit = parent_candidates.begin(),
                                             spcend = parent_candidates.end();
               spcit != spcend; spcit++) {
            if (connected_keyframes[i]->id_ == (*spcit)->id_) {
              int w = keyframe->GetWeight(connected_keyframes[i]);
              if (w > max) {
                child = keyframe;
                parent = connected_keyframes[i];
                max = w;
                do_continue = true;
              }
            }
          }
        }
      }

      if (do_continue) {
        child->ChangeParent(parent);
        parent_candidates.insert(child);
        childrens_.erase(child);
      } else {
        break;
      }
    }

    // If a children has no covisibility links with any parent candidate, assign
    // to the original parent of this KF
    if (!childrens_.empty())
      for (std::set<KeyFrame*>::iterator sit = childrens_.begin();
           sit != childrens_.end(); sit++) {
        (*sit)->ChangeParent(parent_);
      }

    parent_->EraseChild(this);
    Tcp_ = Tcw_ * parent_->GetPoseInverse();
    is_bad_ = true;
  }

  map_->EraseKeyFrame(this);
  keyframe_database_->erase(this);
}

bool KeyFrame::isBad() {
  unique_lock<mutex> lock(mutex_connections_);
  return is_bad_;
}

void KeyFrame::EraseConnection(KeyFrame* keyframe) {
  bool do_update = false;
  {
    unique_lock<mutex> lock(mutex_connections_);
    if (connected_keyframe_weights_.count(keyframe)) {
      connected_keyframe_weights_.erase(keyframe);
      do_update = true;
    }
  }

  if (do_update) {
    UpdateBestCovisibles();
  }
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float& x, const float& y,
                                                const float& r) const {
  std::vector<size_t> indices;
  indices.reserve(N_);

  const int min_cell_x =
      max(0, (int)floor((x - min_x_ - r) * grid_element_width_inv_));
  if (min_cell_x >= grid_cols_) {
    return indices;
  }

  const int max_cell_x =
      min((int)grid_cols_ - 1,
          (int)ceil((x - min_x_ + r) * grid_element_width_inv_));
  if (max_cell_x < 0) {
    return indices;
  }

  const int min_cell_y =
      max(0, (int)floor((y - min_y_ - r) * grid_element_height_inv_));
  if (min_cell_y >= grid_rows_) {
    return indices;
  }

  const int max_cell_y =
      min((int)grid_rows_ - 1,
          (int)ceil((y - min_y_ + r) * grid_element_height_inv_));
  if (max_cell_y < 0) {
    return indices;
  }

  for (int ix = min_cell_x; ix <= max_cell_x; ix++) {
    for (int iy = min_cell_y; iy <= max_cell_y; iy++) {
      const vector<size_t> vCell = grid_[ix][iy];
      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint& kpUn = undistort_keypoints_[vCell[j]];
        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r) {
          indices.push_back(vCell[j]);
        }
      }
    }
  }

  return indices;
}

bool KeyFrame::IsInImage(const float& x, const float& y) const {
  return (x >= min_x_ && x < max_x_ && y >= min_y_ && y < max_y_);
}

float KeyFrame::ComputeSceneMedianDepth(const int q) {
  vector<MapPoint*> map_points;
  Eigen::Matrix4d Tcw;
  {
    unique_lock<mutex> lock(mutex_features_);
    unique_lock<mutex> lock2(mutex_pose_);
    map_points = map_points_;
    Tcw = Tcw_;
  }

  vector<float> depths;
  depths.reserve(N_);
  // Eigen::Vector3d Rcw2 = Tcw.row(2).colRange(0, 3);
  Eigen::Vector3d Rcw2 = Tcw.block<1, 3>(2, 0).transpose();
  Rcw2 = Rcw2.transpose();
  float zcw = Tcw(2, 3);
  for (int i = 0; i < N_; i++) {
    if (map_points_[i]) {
      MapPoint* map_point = map_points_[i];
      Eigen::Vector3d x3Dw = map_point->GetWorldPos();
      float z = Rcw2.dot(x3Dw) + zcw;
      depths.push_back(z);
    }
  }

  sort(depths.begin(), depths.end());

  return depths[(depths.size() - 1) / q];
}

}  // namespace ORB_SLAM2
