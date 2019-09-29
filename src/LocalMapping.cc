/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: LocalMapping.cc
 *
 *          Created On: Wed 04 Sep 2019 04:36:28 PM CST
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

#include "LocalMapping.h"

#include <unistd.h>
#include <mutex>

#include "CeresOptimizer.h"
#include "LoopClosing.h"
#include "MatEigenConverter.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {

LocalMapping::LocalMapping(Map* map, const float is_monocular)
    : is_monocular_(is_monocular),
      is_reset_requested_(false),
      is_finish_requested_(false),
      is_finished_(true),
      map_(map),
      is_abort_BA_(false),
      is_stopped_(false),
      is_stop_requested_(false),
      do_not_stop_(false),
      do_accept_keyframes_(true) {}

void LocalMapping::SetLoopCloser(LoopClosing* loop_closer) {
  loop_closer_ = loop_closer;
}

void LocalMapping::SetTracker(Tracking* tracker) { tracker_ = tracker; }

void LocalMapping::Run() {
  is_finished_ = false;

  while (true) {
    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // BoW conversion and insertion in Map
      ProcessNewKeyFrame();

      // Check recent MapPoints
      MapPointCulling();

      // Triangulate new MapPoints
      CreateNewMapPoints();

      if (!CheckNewKeyFrames()) {
        // Find more matches in neighbor keyframes and fuse point duplications
        SearchInNeighbors();
      }

      is_abort_BA_ = false;

      if (!CheckNewKeyFrames() && !stopRequested()) {
        // Local BA
        if (map_->KeyFramesInMap() > 2)
          CeresOptimizer::LocalBundleAdjustment(current_keyframe_,
                                                &is_abort_BA_, map_);

        // Check redundant local Keyframes
        KeyFrameCulling();
      }

      loop_closer_->InsertKeyFrame(current_keyframe_);
    } else if (Stop()) {
      // Safe area to stop
      while (isStopped() && !CheckFinish()) {
        usleep(3000);
      }
      if (CheckFinish()) break;
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(true);

    if (CheckFinish()) break;

    usleep(3000);
  }

  SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_new_keyframes_);
  new_keyframes_.push_back(keyframe);
  is_abort_BA_ = true;
}

bool LocalMapping::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mutex_new_keyframes_);
  return (!new_keyframes_.empty());
}

void LocalMapping::ProcessNewKeyFrame() {
  {
    unique_lock<mutex> lock(mutex_new_keyframes_);
    current_keyframe_ = new_keyframes_.front();
    new_keyframes_.pop_front();
  }

  // Compute Bags of Words structures
  current_keyframe_->ComputeBoW();

  // Associate MapPoints to the new keyframe and update normal and descriptor
  const vector<MapPoint*> map_point_matches =
      current_keyframe_->GetMapPointMatches();

  for (size_t i = 0; i < map_point_matches.size(); i++) {
    MapPoint* map_point = map_point_matches[i];
    if (map_point) {
      if (!map_point->isBad()) {
        if (!map_point->IsInKeyFrame(current_keyframe_)) {
          map_point->AddObservation(current_keyframe_, i);
          map_point->UpdateNormalAndDepth();
          map_point->ComputeDistinctiveDescriptors();
        } else  // this can only happen for new stereo points inserted by the
                // Tracking
        {
          recent_added_map_points_.push_back(map_point);
        }
      }
    }
  }

  // Update links in the Covisibility Graph
  current_keyframe_->UpdateConnections();

  // Insert Keyframe in Map
  map_->AddKeyFrame(current_keyframe_);
}

void LocalMapping::MapPointCulling() {
  // Check Recent Added MapPoints
  list<MapPoint*>::iterator lit = recent_added_map_points_.begin();
  const unsigned long int current_keyframe_id = current_keyframe_->id_;

  int nThObs = 3;
  const int cnThObs = nThObs;

  while (lit != recent_added_map_points_.end()) {
    MapPoint* map_point = *lit;
    if (map_point->isBad()) {
      lit = recent_added_map_points_.erase(lit);
    } else if (map_point->GetFoundRatio() < 0.25f) {
      map_point->SetBadFlag();
      lit = recent_added_map_points_.erase(lit);
    } else if (((int)current_keyframe_id -
                (int)map_point->first_keyframe_id_) >= 2 &&
               map_point->Observations() <= cnThObs) {
      map_point->SetBadFlag();
      lit = recent_added_map_points_.erase(lit);
    } else if (((int)current_keyframe_id -
                (int)map_point->first_keyframe_id_) >= 3) {
      lit = recent_added_map_points_.erase(lit);
    } else {
      lit++;
    }
  }
}

void LocalMapping::CreateNewMapPoints() {
  // Retrieve neighbor keyframes in covisibility graph
  int nn = 10;
  if (is_monocular_) nn = 20;
  const vector<KeyFrame*> neighbor_keyframes =
      current_keyframe_->GetBestCovisibilityKeyFrames(nn);

  ORBmatcher matcher(0.6, false);

  Eigen::Matrix3d Rcw1 = current_keyframe_->GetRotation();
  Eigen::Matrix3d Rwc1 = Rcw1.transpose();
  Eigen::Vector3d tcw1 = current_keyframe_->GetTranslation();
  Eigen::Matrix4d Tcw1 = Eigen::Matrix4d::Identity();
  Tcw1.block<3, 3>(0, 0) = Rcw1;
  Tcw1.block<3, 1>(0, 3) = tcw1;
  Eigen::Vector3d Ow1 = current_keyframe_->GetCameraCenter();

  const float& fx1 = current_keyframe_->fx_;
  const float& fy1 = current_keyframe_->fy_;
  const float& cx1 = current_keyframe_->cx_;
  const float& cy1 = current_keyframe_->cy_;
  const float& invfx1 = current_keyframe_->invfx_;
  const float& invfy1 = current_keyframe_->invfy_;

  const float ratioFactor = 1.5f * current_keyframe_->scale_factor_;

  int nnew = 0;

  // Search matches with epipolar restriction and triangulate
  for (size_t i = 0; i < neighbor_keyframes.size(); i++) {
    if (i > 0 && CheckNewKeyFrames()) return;

    KeyFrame* neighbor_keyframe = neighbor_keyframes[i];

    // Check first that baseline is not too short
    Eigen::Vector3d Ow2 = neighbor_keyframe->GetCameraCenter();
    Eigen::Vector3d vBaseline = Ow2 - Ow1;
    const float baseline = vBaseline.norm();

    if (!is_monocular_) {
      if (baseline < neighbor_keyframe->mb_) continue;
    } else {
      const float medianDepthKF2 =
          neighbor_keyframe->ComputeSceneMedianDepth(2);
      const float ratioBaselineDepth = baseline / medianDepthKF2;

      if (ratioBaselineDepth < 0.01) continue;
    }

    // Compute Fundamental Matrix
    Eigen::Matrix3d F12 = ComputeF12(current_keyframe_, neighbor_keyframe);

    // Search matches that fullfil epipolar constraint
    vector<pair<size_t, size_t> > matched_indices_;
    matcher.SearchForTriangulation(current_keyframe_, neighbor_keyframe, F12,
                                   matched_indices_, false);

    Eigen::Matrix3d Rcw2 = neighbor_keyframe->GetRotation();
    Eigen::Matrix3d Rwc2 = Rcw2.transpose();
    Eigen::Vector3d tcw2 = neighbor_keyframe->GetTranslation();
    Eigen::Matrix4d Tcw2 = Eigen::Matrix4d::Identity();
    Tcw2.block<3, 3>(0, 0) = Rcw2;
    Tcw2.block<3, 1>(0, 3) = tcw2;

    const float& fx2 = neighbor_keyframe->fx_;
    const float& fy2 = neighbor_keyframe->fy_;
    const float& cx2 = neighbor_keyframe->cx_;
    const float& cy2 = neighbor_keyframe->cy_;
    const float& invfx2 = neighbor_keyframe->invfx_;
    const float& invfy2 = neighbor_keyframe->invfy_;

    // Triangulate each match
    const int nmatches = matched_indices_.size();
    for (int ikp = 0; ikp < nmatches; ikp++) {
      const int& idx1 = matched_indices_[ikp].first;
      const int& idx2 = matched_indices_[ikp].second;

      const cv::KeyPoint& kp1 = current_keyframe_->undistort_keypoints_[idx1];

      const cv::KeyPoint& kp2 = neighbor_keyframe->undistort_keypoints_[idx2];

      // Check parallax between rays
      Eigen::Vector3d xn1;
      xn1 << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0;

      Eigen::Vector3d xn2;
      xn2 << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0;

      Eigen::Vector3d ray1 = Rwc1 * xn1;
      Eigen::Vector3d ray2 = Rwc2 * xn2;
      const float cosParallaxRays =
          ray1.dot(ray2) / (ray1.norm() * ray2.norm());

      float cosParallaxStereo = cosParallaxRays + 1;

      Eigen::Vector4d x4D;
      if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
          cosParallaxRays < 0.9998) {
        // Linear Triangulation Method
        Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
        A.row(0) = xn1[0] * Tcw1.row(2) - Tcw1.row(0);
        A.row(1) = xn1[1] * Tcw1.row(2) - Tcw1.row(1);
        A.row(2) = xn2[0] * Tcw2.row(2) - Tcw2.row(0);
        A.row(3) = xn2[1] * Tcw2.row(2) - Tcw2.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(
            A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        x4D = svd.matrixV().col(3);

        if (x4D[3] == 0) continue;

        // Euclidean coordinates
        x4D = x4D / x4D[3];
      } else {
        continue;  // No stereo and very low parallax
      }

      Eigen::Vector3d x3D = x4D.block<3, 1>(0, 0);

      // Check triangulation in front of cameras
      float z1 = Rcw1.row(2).dot(x3D) + tcw1[2];
      if (z1 <= 0) {
        continue;
      }

      float z2 = Rcw2.row(2).dot(x3D) + tcw2[2];
      if (z2 <= 0) {
        continue;
      }

      // Check reprojection error in first keyframe
      const float& sigmaSquare1 = current_keyframe_->level_sigma2s_[kp1.octave];
      const float x1 = Rcw1.row(0).dot(x3D) + tcw1[0];
      const float y1 = Rcw1.row(1).dot(x3D) + tcw1[1];
      const float invz1 = 1.0 / z1;

      float u1 = fx1 * x1 * invz1 + cx1;
      float v1 = fy1 * y1 * invz1 + cy1;
      float errX1 = u1 - kp1.pt.x;
      float errY1 = v1 - kp1.pt.y;
      if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1) {
        continue;
      }

      // Check reprojection error in second keyframe
      const float sigmaSquare2 = neighbor_keyframe->level_sigma2s_[kp2.octave];
      const float x2 = Rcw2.row(0).dot(x3D) + tcw2[0];
      const float y2 = Rcw2.row(1).dot(x3D) + tcw2[1];
      const float invz2 = 1.0 / z2;

      float u2 = fx2 * x2 * invz2 + cx2;
      float v2 = fy2 * y2 * invz2 + cy2;
      float errX2 = u2 - kp2.pt.x;
      float errY2 = v2 - kp2.pt.y;
      if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2) {
        continue;
      }

      // Check scale consistency
      Eigen::Vector3d normal1 = x3D - Ow1;
      float dist1 = normal1.norm();

      Eigen::Vector3d normal2 = x3D - Ow2;
      float dist2 = normal2.norm();

      if (dist1 == 0 || dist2 == 0) {
        continue;
      }

      const float ratioDist = dist2 / dist1;
      const float ratioOctave = current_keyframe_->scale_factors_[kp1.octave] /
                                neighbor_keyframe->scale_factors_[kp2.octave];

      /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
          continue;*/
      if (ratioDist * ratioFactor < ratioOctave ||
          ratioDist > ratioOctave * ratioFactor) {
        continue;
      }

      // Triangulation is succesfull
      MapPoint* map_point = new MapPoint(x3D, current_keyframe_, map_);

      map_point->AddObservation(current_keyframe_, idx1);
      map_point->AddObservation(neighbor_keyframe, idx2);

      current_keyframe_->AddMapPoint(map_point, idx1);
      neighbor_keyframe->AddMapPoint(map_point, idx2);

      map_point->ComputeDistinctiveDescriptors();

      map_point->UpdateNormalAndDepth();

      map_->AddMapPoint(map_point);
      recent_added_map_points_.push_back(map_point);

      nnew++;
    }
  }
}

void LocalMapping::SearchInNeighbors() {
  // Retrieve neighbor keyframes
  int nn = 10;
  if (is_monocular_) nn = 20;
  const vector<KeyFrame*> neighbor_keyframes =
      current_keyframe_->GetBestCovisibilityKeyFrames(nn);
  vector<KeyFrame*> target_keyframes_;
  for (vector<KeyFrame*>::const_iterator vit = neighbor_keyframes.begin(),
                                         vend = neighbor_keyframes.end();
       vit != vend; vit++) {
    KeyFrame* neighbor_keyframe = *vit;
    if (neighbor_keyframe->isBad() ||
        neighbor_keyframe->fuse_target_for_keyframe_ ==
            current_keyframe_->id_) {
      continue;
    }
    target_keyframes_.push_back(neighbor_keyframe);
    neighbor_keyframe->fuse_target_for_keyframe_ = current_keyframe_->id_;

    // Extend to some second neighbors
    const std::vector<KeyFrame*> second_neighbor_keyframes =
        neighbor_keyframe->GetBestCovisibilityKeyFrames(5);
    for (vector<KeyFrame*>::const_iterator
             vit2 = second_neighbor_keyframes.begin(),
             vend2 = second_neighbor_keyframes.end();
         vit2 != vend2; vit2++) {
      KeyFrame* pKFi2 = *vit2;
      if (pKFi2->isBad() ||
          pKFi2->fuse_target_for_keyframe_ == current_keyframe_->id_ ||
          pKFi2->id_ == current_keyframe_->id_)
        continue;
      target_keyframes_.push_back(pKFi2);
    }
  }

  // Search matches by projection from current KF in target KFs
  ORBmatcher matcher;
  vector<MapPoint*> map_point_matches = current_keyframe_->GetMapPointMatches();
  for (vector<KeyFrame*>::iterator vit = target_keyframes_.begin(),
                                   vend = target_keyframes_.end();
       vit != vend; vit++) {
    KeyFrame* neighbor_keyframe = *vit;

    matcher.Fuse(neighbor_keyframe, map_point_matches);
  }

  // Search matches by projection from target KFs in current KF
  std::vector<MapPoint*> fuse_candidates_;
  fuse_candidates_.reserve(target_keyframes_.size() * map_point_matches.size());

  for (std::vector<KeyFrame*>::iterator vitKF = target_keyframes_.begin(),
                                        vendKF = target_keyframes_.end();
       vitKF != vendKF; vitKF++) {
    KeyFrame* neighbor_keyframe = *vitKF;

    std::vector<MapPoint*> vpMapPointsKFi =
        neighbor_keyframe->GetMapPointMatches();

    for (vector<MapPoint*>::iterator vitMP = vpMapPointsKFi.begin(),
                                     vendMP = vpMapPointsKFi.end();
         vitMP != vendMP; vitMP++) {
      MapPoint* map_point = *vitMP;
      if (!map_point) {
        continue;
      }
      if (map_point->isBad() ||
          map_point->fuse_candidate_for_keyframe == current_keyframe_->id_) {
        continue;
      }
      map_point->fuse_candidate_for_keyframe = current_keyframe_->id_;
      fuse_candidates_.push_back(map_point);
    }
  }

  matcher.Fuse(current_keyframe_, fuse_candidates_);

  // Update points
  map_point_matches = current_keyframe_->GetMapPointMatches();
  for (size_t i = 0, iend = map_point_matches.size(); i < iend; i++) {
    MapPoint* map_point = map_point_matches[i];
    if (map_point) {
      if (!map_point->isBad()) {
        map_point->ComputeDistinctiveDescriptors();
        map_point->UpdateNormalAndDepth();
      }
    }
  }

  // Update connections in covisibility graph
  current_keyframe_->UpdateConnections();
}

Eigen::Matrix3d LocalMapping::ComputeF12(KeyFrame*& pKF1, KeyFrame*& pKF2) {
  Eigen::Matrix3d R1w = pKF1->GetRotation();
  Eigen::Vector3d t1w = pKF1->GetTranslation();
  Eigen::Matrix3d R2w = pKF2->GetRotation();
  Eigen::Vector3d t2w = pKF2->GetTranslation();

  Eigen::Matrix3d R12 = R1w * R2w.transpose();
  Eigen::Vector3d t12 = -R1w * R2w.transpose() * t2w + t1w;

  Eigen::Matrix3d t12x = SkewSymmetricMatrix(t12);

  const Eigen::Matrix3d& K1 = MatEigenConverter::MatToMatrix3d(pKF1->K_);
  const Eigen::Matrix3d& K2 = MatEigenConverter::MatToMatrix3d(pKF2->K_);

  return K1.transpose().inverse() * t12x * R12 * K2.inverse();
}

void LocalMapping::RequestStop() {
  unique_lock<mutex> lock(mutex_stop_);
  is_stop_requested_ = true;
  unique_lock<mutex> lock2(mutex_new_keyframes_);
  is_abort_BA_ = true;
}

bool LocalMapping::Stop() {
  unique_lock<mutex> lock(mutex_stop_);
  if (is_stop_requested_ && !do_not_stop_) {
    is_stopped_ = true;
    cout << "Local Mapping STOP" << endl;
    return true;
  }

  return false;
}

bool LocalMapping::isStopped() {
  unique_lock<mutex> lock(mutex_stop_);
  return is_stopped_;
}

bool LocalMapping::stopRequested() {
  unique_lock<mutex> lock(mutex_stop_);
  return is_stop_requested_;
}

void LocalMapping::Release() {
  unique_lock<mutex> lock(mutex_stop_);
  unique_lock<mutex> lock2(mutex_finish_);
  if (is_finished_) {
    return;
  }
  is_stopped_ = false;
  is_stop_requested_ = false;
  for (list<KeyFrame*>::iterator lit = new_keyframes_.begin(),
                                 lend = new_keyframes_.end();
       lit != lend; lit++)
    delete *lit;
  new_keyframes_.clear();

  cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames() {
  unique_lock<mutex> lock(mutex_accept_);
  return do_accept_keyframes_;
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
  unique_lock<mutex> lock(mutex_accept_);
  do_accept_keyframes_ = flag;
}

bool LocalMapping::SetNotStop(bool flag) {
  unique_lock<mutex> lock(mutex_stop_);

  if (flag && is_stopped_) {
    return false;
  }

  do_not_stop_ = flag;

  return true;
}

void LocalMapping::InterruptBA() { is_abort_BA_ = true; }

void LocalMapping::KeyFrameCulling() {
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are
  // seen in at least other 3 keyframes (in the same or finer scale) We only
  // consider close stereo points
  vector<KeyFrame*> vpLocalKeyFrames =
      current_keyframe_->GetVectorCovisibleKeyFrames();

  for (vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(),
                                   vend = vpLocalKeyFrames.end();
       vit != vend; vit++) {
    KeyFrame* keyframe = *vit;
    if (keyframe->id_ == 0) continue;
    const vector<MapPoint*> map_points = keyframe->GetMapPointMatches();

    int n_observations = 3;
    const int thObs = n_observations;
    int nRedundantObservations = 0;
    int n_map_points = 0;
    for (size_t i = 0, iend = map_points.size(); i < iend; i++) {
      MapPoint* map_point = map_points[i];
      if (map_point) {
        if (!map_point->isBad()) {
          if (!is_monocular_) {
            if (keyframe->depthes_[i] > keyframe->threshold_depth_ ||
                keyframe->depthes_[i] < 0)
              continue;
          }

          n_map_points++;
          if (map_point->Observations() > thObs) {
            const int& scaleLevel = keyframe->undistort_keypoints_[i].octave;
            const map<KeyFrame*, size_t> observations =
                map_point->GetObservations();
            int n_observations = 0;
            for (map<KeyFrame*, size_t>::const_iterator
                     mit = observations.begin(),
                     mend = observations.end();
                 mit != mend; mit++) {
              KeyFrame* neighbor_keyframe = mit->first;
              if (neighbor_keyframe == keyframe) continue;
              const int& scaleLeveli =
                  neighbor_keyframe->undistort_keypoints_[mit->second].octave;

              if (scaleLeveli <= scaleLevel + 1) {
                n_observations++;
                if (n_observations >= thObs) break;
              }
            }
            if (n_observations >= thObs) {
              nRedundantObservations++;
            }
          }
        }
      }
    }

    if (nRedundantObservations > 0.9 * n_map_points) {
      keyframe->SetBadFlag();
    }
  }
}

Eigen::Matrix3d LocalMapping::SkewSymmetricMatrix(const Eigen::Vector3d& v) {
  Eigen::Matrix3d skew;
  skew <<    0, -v[2], v[1],
          v[2],   0  , -v[0],
         -v[1],  v[0], 0;
  return skew;
}

void LocalMapping::RequestReset() {
  {
    unique_lock<mutex> lock(mutex_reset_);
    is_reset_requested_ = true;
  }

  while (true) {
    {
      unique_lock<mutex> lock2(mutex_reset_);
      if (!is_reset_requested_) {
        break;
      }
    }
    usleep(3000);
  }
}

void LocalMapping::ResetIfRequested() {
  unique_lock<mutex> lock(mutex_reset_);
  if (is_reset_requested_) {
    new_keyframes_.clear();
    recent_added_map_points_.clear();
    is_reset_requested_ = false;
  }
}

void LocalMapping::RequestFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  is_finish_requested_ = true;
}

bool LocalMapping::CheckFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  return is_finish_requested_;
}

void LocalMapping::SetFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  is_finished_ = true;
  unique_lock<mutex> lock2(mutex_stop_);
  is_stopped_ = true;
}

bool LocalMapping::isFinished() {
  unique_lock<mutex> lock(mutex_finish_);
  return is_finished_;
}
}  // namespace ORB_SLAM2
