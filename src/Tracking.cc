/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Tracking.cc
 *
 *          Created On: Tue 9 Sep 2019 03:51:14 PM CST
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

#include "Tracking.h"

#include <glog/logging.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "FrameDrawer.h"
#include "Initializer.h"
#include "Map.h"
#include "ORBmatcher.h"

#include "CeresOptimizer.h"
#include "PnPsolver.h"

namespace ORB_SLAM2 {

Tracking::Tracking(MonoORBSlam* mono_orb_slam, ORBVocabulary* vocabulary,
                   FrameDrawer* frame_drawer, MapDrawer* map_drawer, Map* map,
                   KeyFrameDatabase* keyframe_database,
                   const string& string_setting_file)
    : state_(NO_IMAGES_YET),
      do_only_tracking_(false),
      do_vo_(false),
      orb_vocabulary_(vocabulary),
      keyframe_database_(keyframe_database),
      initializer_(static_cast<Initializer*>(nullptr)),
      mono_orb_slam_(mono_orb_slam),
      viewer_(nullptr),
      frame_drawer_(frame_drawer),
      map_drawer_(map_drawer),
      map_(map),
      last_reloc_frame_id_(0) {
  cv::FileStorage fSettings(string_setting_file, cv::FileStorage::READ);
  // Load camera parameters from settings file
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(K_);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(dist_coef_);

  bf_ = fSettings["Camera.bf"];

  float fps = fSettings["Camera.fps"];
  if (fps == 0) fps = 30;

  // Max/Min Frames to insert keyframes and to check relocalisation
  min_frames_ = 0;
  max_frames_ = fps;

  std::cout << std::endl << "Camera Parameters: " << std::endl;
  std::cout << "- fx: " << fx << std::endl;
  std::cout << "- fy: " << fy << std::endl;
  std::cout << "- cx: " << cx << std::endl;
  std::cout << "- cy: " << cy << std::endl;
  std::cout << "- k1: " << DistCoef.at<float>(0) << std::endl;
  std::cout << "- k2: " << DistCoef.at<float>(1) << std::endl;
  if (DistCoef.rows == 5)
    std::cout << "- k3: " << DistCoef.at<float>(4) << std::endl;
  std::cout << "- p1: " << DistCoef.at<float>(2) << std::endl;
  std::cout << "- p2: " << DistCoef.at<float>(3) << std::endl;
  std::cout << "- fps: " << fps << std::endl;

  int nRGB = fSettings["Camera.RGB"];
  is_rgb_ = nRGB;

  if (is_rgb_)
    std::cout << "- color order: RGB (ignored if grayscale)" << std::endl;
  else
    std::cout << "- color order: BGR (ignored if grayscale)" << std::endl;

  // Load ORB parameters

  int n_features = fSettings["ORBextractor.nFeatures"];
  float scale_factor = fSettings["ORBextractor.scaleFactor"];
  int n_levels = fSettings["ORBextractor.nLevels"];
  int init_th_FAST = fSettings["ORBextractor.iniThFAST"];
  int min_th_FAST = fSettings["ORBextractor.minThFAST"];

  orb_extractor_left_ = new ORBextractor(n_features, scale_factor, n_levels,
                                         init_th_FAST, min_th_FAST);

  init_orb_extractor_ = new ORBextractor(2 * n_features, scale_factor, n_levels,
                                         init_th_FAST, min_th_FAST);

  std::cout << std::endl << "ORB Extractor Parameters: " << std::endl;
  std::cout << "- Number of Features: " << n_features << std::endl;
  std::cout << "- Scale Levels: " << n_levels << std::endl;
  std::cout << "- Scale Factor: " << scale_factor << std::endl;
  std::cout << "- Initial Fast Threshold: " << init_th_FAST << std::endl;
  std::cout << "- Minimum Fast Threshold: " << min_th_FAST << std::endl;
}

void Tracking::SetLocalMapper(LocalMapping* local_mapper) {
  local_mapper_ = local_mapper;
}

void Tracking::SetLoopClosing(LoopClosing* loop_closing) {
  loop_closing_ = loop_closing;
}

void Tracking::SetViewer(Viewer* viewer) { viewer_ = viewer; }

Eigen::Matrix4d Tracking::GrabImageMonocular(const cv::Mat& img,
                                             const double& timestamp) {
  // static int frame_num = 0;
  img_gray_ = img;

  if (img_gray_.channels() == 3) {
    if (is_rgb_)
      cvtColor(img_gray_, img_gray_, CV_RGB2GRAY);
    else
      cvtColor(img_gray_, img_gray_, CV_BGR2GRAY);
  } else if (img_gray_.channels() == 4) {
    if (is_rgb_)
      cvtColor(img_gray_, img_gray_, CV_RGBA2GRAY);
    else
      cvtColor(img_gray_, img_gray_, CV_BGRA2GRAY);
  }

  if (state_ == NOT_INITIALIZED || state_ == NO_IMAGES_YET)
    current_frame_ =
        Frame(img_gray_, timestamp, init_orb_extractor_, orb_vocabulary_, K_,
              dist_coef_, bf_, theshold_depth_);
  else
    current_frame_ =
        Frame(img_gray_, timestamp, orb_extractor_left_, orb_vocabulary_, K_,
              dist_coef_, bf_, theshold_depth_);

  Track();

  return current_frame_.Tcw_;
}

bool Tracking::TrackingWithKnownMap() {
  // Localization Mode: Local Mapping is deactivated
  bool is_OK = false;
  if (state_ == LOST) {
      is_OK = Relocalization();
  } else {
    if (!do_vo_) {
      // In last frame we tracked enough MapPoints in the map
      if (!velocity_.isIdentity()) {
        is_OK = TrackWithMotionModel();
      } else {
        is_OK = TrackReferenceKeyFrame();
      }
    } else {
      // In last frame we tracked mainly "visual odometry" points.

      // We compute two camera poses, one from motion model and one
      // doing relocalization. If relocalization is sucessfull we choose
      // that solution, otherwise we retain the "visual odometry"
      // solution.
      bool is_motion_model_OK = false;
      bool is_reloc_OK = false;
      vector<MapPoint*> motion_model_map_points;
      vector<bool> is_motion_model_outliers;
      Eigen::Matrix4d motion_model_Tcw;
      if (!velocity_.isIdentity()) {
        is_motion_model_OK = TrackWithMotionModel();
        motion_model_map_points = current_frame_.map_points_;
        is_motion_model_outliers = current_frame_.is_outliers_;
        motion_model_Tcw = current_frame_.Tcw_;
      }
      is_reloc_OK = Relocalization();

      if (is_motion_model_OK && !is_reloc_OK) {
        current_frame_.SetPose(motion_model_Tcw);
        current_frame_.map_points_ = motion_model_map_points;
        current_frame_.is_outliers_ = is_motion_model_outliers;

        if (do_vo_) {
          for (int i = 0; i < current_frame_.N_; i++) {
            if (current_frame_.map_points_[i] &&
                !current_frame_.is_outliers_[i]) {
              current_frame_.map_points_[i]->IncreaseFound();
            }
          }
        }
      } else if (is_reloc_OK) {
        do_vo_ = false;
      }
      is_OK = is_reloc_OK || is_motion_model_OK;
    }
  }
  return is_OK;
}

bool Tracking::Mapping() {
  bool is_OK = false;
  // Local Mapping is activated. This is the normal behaviour, unless
  // you explicitly activate the "only tracking" mode.
  if (state_ == OK) {
    // Local Mapping might have changed some MapPoints tracked in last
    // frame
    CheckReplacedInLastFrame();

    if (velocity_.isIdentity() ||
        current_frame_.id_ < last_reloc_frame_id_ + 2) {
      is_OK = TrackReferenceKeyFrame();
    } else {
      is_OK = TrackWithMotionModel();
      if (!is_OK) {
        is_OK = TrackReferenceKeyFrame();
      }
    }

  } else {
    is_OK = Relocalization();
  }
  return is_OK;
}

void Tracking::Track() {
  if (state_ == NO_IMAGES_YET) {
    state_ = NOT_INITIALIZED;
  }
  last_processed_state_ = state_;

  // Get Map Mutex -> Map cannot be changed
  unique_lock<mutex> lock(map_->mutex_map_update_);

  if (state_ == NOT_INITIALIZED) {
    MonocularInitialization();
    frame_drawer_->Update(this);
    if (state_ != OK) {
      return;
    }
    viewer_->SetFollowCamera();
  } else {
    // System is initialized. Track Frame.
    bool is_OK;
    // Initial camera pose estimation using motion model or relocalization (if
    // tracking is lost)
    if (!do_only_tracking_) {
      is_OK = Mapping();
      current_frame_.reference_keyframe_ = reference_keyframe_;
      // If we have an initial estimation of the camera pose and matching.
      // Track the local map.
      if (is_OK) {
        is_OK = TrackLocalMap();
      }
    } else {
      is_OK = TrackingWithKnownMap();
      current_frame_.reference_keyframe_ = reference_keyframe_;
      // do_vo_ true means that there are few matches to MapPoints in the
      // map. We cannot retrieve a local map and therefore we do not perform
      // TrackLocalMap(). Once the system relocalizes the camera we will use
      // the local map again.
      if (is_OK && !do_vo_) {
        is_OK = TrackLocalMap();
      }
    } // end of if (!do_only_tracking_)

    state_ = is_OK ? OK : LOST;

    // Update drawer
    frame_drawer_->Update(this);

    // If tracking were good, check if we insert a keyframe
    if (is_OK) {
      // Update motion model
      if (!last_frame_.Tcw_.isIdentity()) {
        Eigen::Matrix4d last_Twc = Eigen::Matrix4d::Identity();
        last_Twc.block<3, 3>(0, 0) = last_frame_.GetRotationInverse();
        last_Twc.block<3, 1>(0, 3) = last_frame_.GetCameraCenter();
        velocity_ = current_frame_.Tcw_ * last_Twc;
      } else {
        velocity_ = Eigen::Matrix4d::Identity();
      }

      map_drawer_->SetCurrentCameraPose(current_frame_.Tcw_);

      // Clean VO matches
      for (int i = 0; i < current_frame_.N_; i++) {
        MapPoint* map_point = current_frame_.map_points_[i];
        if (map_point)
          if (map_point->Observations() < 1) {
            current_frame_.is_outliers_[i] = false;
            current_frame_.map_points_[i] = static_cast<MapPoint*>(nullptr);
          }
      }

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame()) {
        CreateNewKeyFrame();
      }

      // We allow points with high innovation (considererd outliers by the
      // Huber Function) pass to the new keyframe, so that bundle adjustment
      // will finally decide if they are outliers or not. We don't want next
      // frame to estimate its position with those points so we discard them
      // in the frame.
      for (int i = 0; i < current_frame_.N_; i++) {
        if (current_frame_.map_points_[i] && current_frame_.is_outliers_[i])
          current_frame_.map_points_[i] = static_cast<MapPoint*>(nullptr);
      }
    }

    // Reset if the camera get lost soon after initialization
    if (state_ == LOST) {
      if (map_->KeyFramesInMap() <= 5) {
        std::cout << "Track lost soon after initialisation, reseting..."
                  << std::endl;
        mono_orb_slam_->Reset();
        return;
      }
    }

    if (!current_frame_.reference_keyframe_)
      current_frame_.reference_keyframe_ = reference_keyframe_;

    last_frame_ = Frame(current_frame_);
  }

  // Store frame pose information to retrieve the complete camera trajectory
  // afterwards.
  if (!current_frame_.Tcw_.isIdentity()) {
    Eigen::Matrix4d Tcr = current_frame_.Tcw_ *
                          current_frame_.reference_keyframe_->GetPoseInverse();
    relative_frame_poses_.push_back(Tcr);
    reference_keyframes_.push_back(reference_keyframe_);
    frame_times_.push_back(current_frame_.timestamp_);
    do_lostes_.push_back(state_ == LOST);
  } else {
    // This can happen if tracking is lost
    relative_frame_poses_.push_back(relative_frame_poses_.back());
    reference_keyframes_.push_back(reference_keyframes_.back());
    frame_times_.push_back(frame_times_.back());
    do_lostes_.push_back(state_ == LOST);
  }
}

void Tracking::MonocularInitialization() {
  if (!initializer_) {
    // Set Reference Frame
    if (current_frame_.keypoints_.size() > 100) {
      init_frame_ = Frame(current_frame_);
      last_frame_ = Frame(current_frame_);
      pre_matched_keypoints_.resize(current_frame_.undistort_keypoints_.size());
      for (size_t i = 0; i < current_frame_.undistort_keypoints_.size(); i++)
        pre_matched_keypoints_[i] = current_frame_.undistort_keypoints_[i].pt;

      if (initializer_) {
        delete initializer_;
      }

      initializer_ = new Initializer(current_frame_, 1.0, 200);

      std::fill(init_matches_.begin(), init_matches_.end(), -1);

      return;
    }
  } else {
    // Try to initialize
    if ((int)current_frame_.keypoints_.size() <= 100) {
      delete initializer_;
      initializer_ = static_cast<Initializer*>(nullptr);
      std::fill(init_matches_.begin(), init_matches_.end(), -1);
      return;
    }

    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(init_frame_, current_frame_,
                                                   pre_matched_keypoints_,
                                                   init_matches_, 100);

    // Check if there are enough correspondences
    if (nmatches < 100) {
      delete initializer_;
      initializer_ = static_cast<Initializer*>(nullptr);
      return;
    }

    Eigen::Matrix3d Rcw;  // Current Camera Rotation
    Eigen::Vector3d tcw;  // Current Camera Translation
    // Triangulated Correspondences (init_matches_)
    vector<bool> is_triangulated;

    if (initializer_->Initialize(current_frame_, init_matches_, Rcw, tcw,
                                 init_P3Ds_, is_triangulated)) {
      for (size_t i = 0, iend = init_matches_.size(); i < iend; i++) {
        if (init_matches_[i] >= 0 && !is_triangulated[i]) {
          init_matches_[i] = -1;
          nmatches--;
        }
      }

      // Set Frame Poses
      init_frame_.SetPose(Eigen::Matrix4d::Identity());

      Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
      Tcw.block<3, 3>(0, 0) = Rcw;
      Tcw.block<3, 1>(0, 3) = tcw;

      current_frame_.SetPose(Tcw);

      CreateInitialMapMonocular();
    }
  }
}

void Tracking::CreateInitialMapMonocular() {
  // Create KeyFrames
  KeyFrame* init_keyframe = new KeyFrame(init_frame_, map_, keyframe_database_);
  KeyFrame* current_keyframe =
      new KeyFrame(current_frame_, map_, keyframe_database_);

  init_keyframe->ComputeBoW();
  current_keyframe->ComputeBoW();

  // Insert KFs in the map
  map_->AddKeyFrame(init_keyframe);
  map_->AddKeyFrame(current_keyframe);

  // Create MapPoints and asscoiate to keyframes
  for (size_t i = 0; i < init_matches_.size(); i++) {
    if (init_matches_[i] < 0) continue;

    // Create MapPoint.
    Eigen::Vector3d worldPos = init_P3Ds_[i];

    MapPoint* map_point = new MapPoint(worldPos, current_keyframe, map_);

    init_keyframe->AddMapPoint(map_point, i);
    current_keyframe->AddMapPoint(map_point, init_matches_[i]);

    map_point->AddObservation(init_keyframe, i);
    map_point->AddObservation(current_keyframe, init_matches_[i]);

    map_point->ComputeDistinctiveDescriptors();
    map_point->UpdateNormalAndDepth();

    // Fill Current Frame structure
    current_frame_.map_points_[init_matches_[i]] = map_point;
    current_frame_.is_outliers_[init_matches_[i]] = false;

    // Add to Map
    map_->AddMapPoint(map_point);
  }

  // Update Connections
  init_keyframe->UpdateConnections();
  current_keyframe->UpdateConnections();

  // Bundle Adjustment
  LOG(INFO) << "New Map created with " << map_->MapPointsInMap() << " points";

  // Optimizer::GlobalBundleAdjustemnt(map_, 20);
  CeresOptimizer::GlobalBundleAdjustemnt(map_, 20);

  // Set median depth to 1
  float median_depth = init_keyframe->ComputeSceneMedianDepth(2);
  float inv_median_depth = 1.0f / median_depth;

  if (median_depth < 0 || current_keyframe->TrackedMapPoints(1) < 80) {
    std::cout << "Wrong initialization, reseting..." << std::endl;
    Reset();
    return;
  }

  // Scale initial baseline
  Eigen::Matrix4d Tc2w = current_keyframe->GetPose();
  // Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * inv_median_depth;
  Tc2w.block<3, 1>(0, 3) = Tc2w.block<3, 1>(0, 3) * inv_median_depth;
  current_keyframe->SetPose(Tc2w);

  // Scale points
  std::vector<MapPoint*> all_map_points = init_keyframe->GetMapPointMatches();
  for (size_t iMP = 0; iMP < all_map_points.size(); iMP++) {
    if (all_map_points[iMP]) {
      MapPoint* map_point = all_map_points[iMP];
      map_point->SetWorldPos(map_point->GetWorldPos() * inv_median_depth);
    }
  }

  local_mapper_->InsertKeyFrame(init_keyframe);
  local_mapper_->InsertKeyFrame(current_keyframe);

  current_frame_.SetPose(current_keyframe->GetPose());
  last_keyframe_id_ = current_frame_.id_;
  last_keyframe_ = current_keyframe;

  local_keyframes_.push_back(current_keyframe);
  local_keyframes_.push_back(init_keyframe);
  local_map_points_ = map_->GetAllMapPoints();
  reference_keyframe_ = current_keyframe;
  current_frame_.reference_keyframe_ = current_keyframe;

  last_frame_ = Frame(current_frame_);

  map_->SetReferenceMapPoints(local_map_points_);

  map_drawer_->SetCurrentCameraPose(current_keyframe->GetPose());

  map_->keyframe_origins_.push_back(init_keyframe);

  state_ = OK;
}

void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < last_frame_.N_; i++) {
    MapPoint* map_point = last_frame_.map_points_[i];

    if (map_point) {
      MapPoint* replaced_map_point = map_point->GetReplaced();
      if (replaced_map_point) {
        last_frame_.map_points_[i] = replaced_map_point;
      }
    }
  }
}

bool Tracking::TrackReferenceKeyFrame() {
  // LOG(WARNING) << "Track ReferenceKeyFrame";
  // Compute Bag of Words vector
  current_frame_.ComputeBoW();

  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.7, true);
  vector<MapPoint*> matched_map_points;

  int nmatches = matcher.SearchByBoW(reference_keyframe_, current_frame_,
                                     matched_map_points);

  // LOG(INFO) << "TrackReferenceKeyFrame nmatches: " << nmatches;
  if (nmatches < 15) {
    return false;
  }

  current_frame_.map_points_ = matched_map_points;
  current_frame_.SetPose(last_frame_.Tcw_);

  CeresOptimizer::PoseOptimization(&current_frame_);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < current_frame_.N_; i++) {
    if (current_frame_.map_points_[i]) {
      if (current_frame_.is_outliers_[i]) {
        MapPoint* map_point = current_frame_.map_points_[i];

        current_frame_.map_points_[i] = static_cast<MapPoint*>(nullptr);
        current_frame_.is_outliers_[i] = false;
        map_point->is_track_in_view_ = false;
        map_point->last_seen_frame_id_ = current_frame_.id_;
        nmatches--;
      } else if (current_frame_.map_points_[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame* reference_keyframe = last_frame_.reference_keyframe_;
  Eigen::Matrix4d Tlr = relative_frame_poses_.back();

  last_frame_.SetPose(Tlr * reference_keyframe->GetPose());
}

bool Tracking::TrackWithMotionModel() {
  // LOG(WARNING) << "Track With Motion Model";
  ORBmatcher matcher(0.9, true);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();

  current_frame_.SetPose(velocity_ * last_frame_.Tcw_);

  std::fill(current_frame_.map_points_.begin(),
            current_frame_.map_points_.end(), static_cast<MapPoint*>(nullptr));

  // Project points seen in previous frame
  int th = 15;
  int nmatches = matcher.SearchByProjection(current_frame_, last_frame_, th);

  // If few matches, uses a wider window search
  if (nmatches < 20) {
    std::fill(current_frame_.map_points_.begin(),
              current_frame_.map_points_.end(), static_cast<MapPoint*>(nullptr));
    nmatches = matcher.SearchByProjection(current_frame_, last_frame_, 2 * th);
  }

  if (nmatches < 20) {
    return false;
  }

  // Optimize frame pose with all matches
  CeresOptimizer::PoseOptimization(&current_frame_);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < current_frame_.N_; i++) {
    if (current_frame_.map_points_[i]) {
      if (current_frame_.is_outliers_[i]) {
        MapPoint* map_point = current_frame_.map_points_[i];

        current_frame_.map_points_[i] = static_cast<MapPoint*>(nullptr);
        current_frame_.is_outliers_[i] = false;
        map_point->is_track_in_view_ = false;
        map_point->last_seen_frame_id_ = current_frame_.id_;
        nmatches--;
      } else if (current_frame_.map_points_[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (do_only_tracking_) {
    do_vo_ = nmatchesMap < 10;
    return nmatches > 20;
  }

  return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
  // LOG(WARNING) << "Track Local Map";
  // We have an estimation of the camera pose and some map points tracked in
  // the frame. We retrieve the local map and try to find matches to points in
  // the local map.

  UpdateLocalMap();

  SearchLocalPoints();

  // Optimize Pose
  CeresOptimizer::PoseOptimization(&current_frame_);
  n_matches_inliers_ = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < current_frame_.N_; i++) {
    if (current_frame_.map_points_[i]) {
      if (!current_frame_.is_outliers_[i]) {
        current_frame_.map_points_[i]->IncreaseFound();
        if (!do_only_tracking_) {
          if (current_frame_.map_points_[i]->Observations() > 0) {
            n_matches_inliers_++;
          }
        } else {
          n_matches_inliers_++;
        }
      }
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (current_frame_.id_ < last_reloc_frame_id_ + max_frames_ &&
      n_matches_inliers_ < 50) {
    return false;
  }

  if (n_matches_inliers_ < 30) {
    return false;
  } else {
    return true;
  }
}

bool Tracking::NeedNewKeyFrame() {
  if (do_only_tracking_) {
    return false;
  }

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (local_mapper_->isStopped() || local_mapper_->stopRequested())
    return false;

  const int n_keyframes = map_->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last
  // relocalisation
  if (current_frame_.id_ < last_reloc_frame_id_ + max_frames_ &&
      n_keyframes > max_frames_)
    return false;

  // Tracked MapPoints in the reference keyframe
  int n_min_observations = 3;
  if (n_keyframes <= 2) {
    n_min_observations = 2;
  }
  int nRefMatches = reference_keyframe_->TrackedMapPoints(n_min_observations);

  // Local Mapping accept keyframes?
  bool is_local_mapping_idle = local_mapper_->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be
  // potentially created.

  // Thresholds
  float thRefRatio = 0.9f;

  // Condition 1a: More than "MaxFrames" have passed from last keyframe
  // insertion
  const bool c1a = current_frame_.id_ >= last_keyframe_id_ + max_frames_;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (current_frame_.id_ >= last_keyframe_id_ + min_frames_ &&
                    is_local_mapping_idle);
  // Condition 1c: tracking is weak
  const bool c1c = false;
  // Condition 2: Few tracked points compared to reference keyframe. Lots of
  // visual odometry compared to map matches.
  const bool c2 = ((n_matches_inliers_ < nRefMatches * thRefRatio) &&
                   n_matches_inliers_ > 15);

  if ((c1a || c1b || c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (is_local_mapping_idle) {
      return true;
    } else {
      local_mapper_->InterruptBA();
      return false;
    }
  } else {
    return false;
  }
}

void Tracking::CreateNewKeyFrame() {
  if (!local_mapper_->SetNotStop(true)) return;

  KeyFrame* keyframe = new KeyFrame(current_frame_, map_, keyframe_database_);

  reference_keyframe_ = keyframe;
  current_frame_.reference_keyframe_ = keyframe;

  local_mapper_->InsertKeyFrame(keyframe);

  local_mapper_->SetNotStop(false);

  last_keyframe_id_ = current_frame_.id_;
  last_keyframe_ = keyframe;
}

void Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  for (std::vector<MapPoint*>::iterator
           vit = current_frame_.map_points_.begin(),
           vend = current_frame_.map_points_.end();
       vit != vend; vit++) {
    MapPoint* map_point = *vit;
    if (map_point) {
      if (map_point->isBad()) {
        *vit = static_cast<MapPoint*>(nullptr);
      } else {
        map_point->IncreaseVisible();
        map_point->last_seen_frame_id_ = current_frame_.id_;
        map_point->is_track_in_view_ = false;
      }
    }
  }

  int n_to_match = 0;

  // Project points in frame and check its visibility
  for (std::vector<MapPoint*>::iterator vit = local_map_points_.begin(),
                                        vend = local_map_points_.end();
       vit != vend; vit++) {
    MapPoint* map_point = *vit;
    if (map_point->last_seen_frame_id_ == current_frame_.id_) continue;
    if (map_point->isBad()) continue;
    // Project (this fills MapPoint variables for matching)
    if (current_frame_.isInFrustum(map_point, 0.5)) {
      map_point->IncreaseVisible();
      n_to_match++;
    }
  }

  if (n_to_match > 0) {
    ORBmatcher matcher(0.8);
    int th = 1;
    // If the camera has been relocalised recently, perform a coarser search
    if (current_frame_.id_ < last_reloc_frame_id_ + 2) {
      th = 5;
    }
    matcher.SearchByProjection(current_frame_, local_map_points_, th);
  }
}

void Tracking::UpdateLocalMap() {
  // This is for visualization
  map_->SetReferenceMapPoints(local_map_points_);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
  local_map_points_.clear();

  for (std::vector<KeyFrame*>::const_iterator itKF = local_keyframes_.begin(),
                                              itEndKF = local_keyframes_.end();
       itKF != itEndKF; itKF++) {
    KeyFrame* keyframe = *itKF;
    const vector<MapPoint*> map_point_matches = keyframe->GetMapPointMatches();

    for (vector<MapPoint*>::const_iterator itMP = map_point_matches.begin(),
                                           itEndMP = map_point_matches.end();
         itMP != itEndMP; itMP++) {
      MapPoint* map_point = *itMP;
      if (!map_point) {
        continue;
      }
      if (map_point->track_reference_for_frame_ == current_frame_.id_) {
        continue;
      }
      if (!map_point->isBad()) {
        local_map_points_.push_back(map_point);
        map_point->track_reference_for_frame_ = current_frame_.id_;
      }
    }
  }
}

void Tracking::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  std::map<KeyFrame*, int> keyframeCounter;
  for (int i = 0; i < current_frame_.N_; i++) {
    if (current_frame_.map_points_[i]) {
      MapPoint* map_point = current_frame_.map_points_[i];
      if (!map_point->isBad()) {
        const std::map<KeyFrame*, size_t> observations =
            map_point->GetObservations();
        for (std::map<KeyFrame*, size_t>::const_iterator
                 it = observations.begin(),
                 itend = observations.end();
             it != itend; it++)
          keyframeCounter[it->first]++;
      } else {
        current_frame_.map_points_[i] = nullptr;
      }
    }
  }

  if (keyframeCounter.empty()) {
    return;
  }

  int max = 0;
  KeyFrame* keyframe_max = static_cast<KeyFrame*>(nullptr);

  local_keyframes_.clear();
  local_keyframes_.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map.
  // Also check which keyframe shares most points
  for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(),
                                           itEnd = keyframeCounter.end();
       it != itEnd; it++) {
    KeyFrame* keyframe = it->first;

    if (keyframe->isBad()) continue;

    if (it->second > max) {
      max = it->second;
      keyframe_max = keyframe;
    }

    local_keyframes_.push_back(it->first);
    keyframe->track_reference_for_frame_ = current_frame_.id_;
  }

  // Include also some not-already-included keyframes that are neighbors to
  // already-included keyframes
  for (vector<KeyFrame*>::const_iterator itKF = local_keyframes_.begin(),
                                         itEndKF = local_keyframes_.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (local_keyframes_.size() > 80) break;

    KeyFrame* keyframe = *itKF;

    const vector<KeyFrame*> neighbor_keyframes =
        keyframe->GetBestCovisibilityKeyFrames(10);

    for (vector<KeyFrame*>::const_iterator it = neighbor_keyframes.begin(),
                                           itend = neighbor_keyframes.end();
         it != itend; it++) {
      KeyFrame* neighbor_keyframe = *it;
      if (!neighbor_keyframe->isBad()) {
        if (neighbor_keyframe->track_reference_for_frame_ !=
            current_frame_.id_) {
          local_keyframes_.push_back(neighbor_keyframe);
          neighbor_keyframe->track_reference_for_frame_ = current_frame_.id_;
          break;
        }
      }
    }

    const std::set<KeyFrame*> childrens = keyframe->GetChilds();
    for (std::set<KeyFrame*>::const_iterator sit = childrens.begin(),
                                             send = childrens.end();
         sit != send; sit++) {
      KeyFrame* child_keyframe = *sit;
      if (!child_keyframe->isBad()) {
        if (child_keyframe->track_reference_for_frame_ != current_frame_.id_) {
          local_keyframes_.push_back(child_keyframe);
          child_keyframe->track_reference_for_frame_ = current_frame_.id_;
          break;
        }
      }
    }

    KeyFrame* parent = keyframe->GetParent();
    if (parent) {
      if (parent->track_reference_for_frame_ != current_frame_.id_) {
        local_keyframes_.push_back(parent);
        parent->track_reference_for_frame_ = current_frame_.id_;
        break;
      }
    }
  }

  if (keyframe_max) {
    reference_keyframe_ = keyframe_max;
    current_frame_.reference_keyframe_ = reference_keyframe_;
  }
}

bool Tracking::Relocalization() {
  // LOG(WARNING) << " Relocalization Enter ";
  // Compute Bag of Words Vector
  current_frame_.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for
  // relocalisation
  vector<KeyFrame*> candidate_keyframes =
      keyframe_database_->DetectRelocalizationCandidates(&current_frame_);

  // LOG(INFO) << " Relocalization candidate keyframe size: "
  //          << candidate_keyframes.size();

  if (candidate_keyframes.empty()) {
    return false;
  }

  const int n_keyframes = candidate_keyframes.size();

  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.75, true);

  vector<PnPsolver*> PnP_solvers;
  PnP_solvers.resize(n_keyframes);

  vector<vector<MapPoint*> > map_point_matches_vector;
  map_point_matches_vector.resize(n_keyframes);

  vector<bool> do_discarded;
  do_discarded.resize(n_keyframes);

  int n_candidates = 0;

  for (int i = 0; i < n_keyframes; i++) {
    KeyFrame* keyframe = candidate_keyframes[i];
    if (keyframe->isBad())
      do_discarded[i] = true;
    else {
      int nmatches = matcher.SearchByBoW(keyframe, current_frame_,
                                         map_point_matches_vector[i]);
      if (nmatches < 15) {
        do_discarded[i] = true;
        continue;
      } else {
        PnPsolver* pSolver =
            new PnPsolver(current_frame_, map_point_matches_vector[i]);
        pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
        PnP_solvers[i] = pSolver;
        n_candidates++;
      }
    }
  }

  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool is_match = false;
  ORBmatcher matcher2(0.9, true);

  while (n_candidates > 0 && !is_match) {
    for (int i = 0; i < n_keyframes; i++) {
      if (do_discarded[i]) continue;

      // Perform 5 Ransac Iterations
      vector<bool> is_inliers;
      int n_inliers;
      bool is_no_more;

      PnPsolver* pSolver = PnP_solvers[i];
      Eigen::Matrix4d Tcw =
          pSolver->iterate(5, is_no_more, is_inliers, n_inliers);

      // If Ransac reachs max. iterations discard keyframe
      if (is_no_more) {
        do_discarded[i] = true;
        n_candidates--;
      }

      // If a Camera Pose is computed, optimize
      if (!Tcw.isIdentity()) {
        current_frame_.Tcw_ = Tcw;

        std::set<MapPoint*> map_points_founded;

        const int np = is_inliers.size();

        for (int j = 0; j < np; j++) {
          if (is_inliers[j]) {
            current_frame_.map_points_[j] = map_point_matches_vector[i][j];
            map_points_founded.insert(map_point_matches_vector[i][j]);
          } else
            current_frame_.map_points_[j] = nullptr;
        }

        int nGood = CeresOptimizer::PoseOptimization(&current_frame_);

        if (nGood < 10) continue;

        for (int io = 0; io < current_frame_.N_; io++)
          if (current_frame_.is_outliers_[io])
            current_frame_.map_points_[io] = static_cast<MapPoint*>(nullptr);

        // If few inliers, search by projection in a coarse window and
        // optimize again
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(
              current_frame_, candidate_keyframes[i], map_points_founded, 10,
              100);

          if (nadditional + nGood >= 50) {
            nGood = CeresOptimizer::PoseOptimization(&current_frame_);

            // If many inliers but still not enough, search by projection
            // again in a narrower window the camera has been already
            // optimized with many points
            if (nGood > 30 && nGood < 50) {
              map_points_founded.clear();
              for (int ip = 0; ip < current_frame_.N_; ip++) {
                if (current_frame_.map_points_[ip]) {
                  map_points_founded.insert(current_frame_.map_points_[ip]);
                }
                nadditional = matcher2.SearchByProjection(
                    current_frame_, candidate_keyframes[i], map_points_founded,
                    3, 64);
              }

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = CeresOptimizer::PoseOptimization(&current_frame_);

                for (int io = 0; io < current_frame_.N_; io++) {
                  if (current_frame_.is_outliers_[io]) {
                    current_frame_.map_points_[io] = nullptr;
                  }
                }
              }
            }
          }
        }

        // If the pose is supported by enough inliers stop ransacs and
        // continue
        if (nGood >= 50) {
          is_match = true;
          break;
        }
      }
    }
  }

  // LOG(INFO) << " Relocalization done, is matched? " << is_match;
  if (!is_match) {
    return false;
  } else {
    last_reloc_frame_id_ = current_frame_.id_;
    return true;
  }
}

void Tracking::Reset() {
  std::cout << "System Reseting" << std::endl;
  if (viewer_) {
    viewer_->RequestStop();
    while (!viewer_->isStopped()) usleep(3000);
  }

  // Reset Local Mapping
  std::cout << "Reseting Local Mapper...";
  local_mapper_->RequestReset();
  std::cout << " done" << std::endl;

  // Reset Loop Closing
  std::cout << "Reseting Loop Closing...";
  loop_closing_->RequestReset();
  std::cout << " done" << std::endl;

  // Clear BoW Database
  std::cout << "Reseting Database...";
  keyframe_database_->clear();
  std::cout << " done" << std::endl;

  // Clear Map (this erase MapPoints and KeyFrames)
  map_->clear();

  KeyFrame::next_id_ = 0;
  Frame::next_id_ = 0;
  state_ = NO_IMAGES_YET;

  if (initializer_) {
    delete initializer_;
    initializer_ = static_cast<Initializer*>(nullptr);
  }

  relative_frame_poses_.clear();
  reference_keyframes_.clear();
  frame_times_.clear();
  do_lostes_.clear();

  if (viewer_) viewer_->Release();
}

void Tracking::InformOnlyTracking(const bool& flag) {
  do_only_tracking_ = flag;
}
}  // namespace ORB_SLAM2
