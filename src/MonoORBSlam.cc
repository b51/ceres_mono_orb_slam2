/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: MonoORBSlam.cc
 *
 *          Created On: Tue 03 Sep 2019 06:27:56 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#include "MonoORBSlam.h"

#include <pangolin/pangolin.h>
#include <unistd.h>
#include <iomanip>
#include <thread>

namespace ORB_SLAM2 {

MonoORBSlam::MonoORBSlam(const string& voc_file,
                         const string& string_setting_file,
                         const bool is_use_viewer)
    : do_reset_(false),
      is_activate_localization_mode_(false),
      is_deactivate_localization_mode_(false) {
  std::cout
      << std::endl
      << "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of "
         "Zaragoza."
      << std::endl
      << "ceres_mono_orb_slam2 Copyright (C) 2019 b51live"
      << std::endl
      << "This program comes with ABSOLUTELY NO WARRANTY;" << std::endl
      << "This is free software, and you are welcome to redistribute it"
      << std::endl
      << "under certain conditions. See LICENSE.txt." << std::endl
      << std::endl;

  // Check settings file
  cv::FileStorage fsSettings(string_setting_file.c_str(),
                             cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    LOG(FATAL) << "Failed to open settings file at: " << string_setting_file;
  }

  // Load ORB Vocabulary
  std::cout << std::endl
            << "Loading ORB Vocabulary. This could take a while..."
            << std::endl;

  vocabulary_ = new ORBVocabulary();
  bool voc_loaded = vocabulary_->loadFromTextFile(voc_file);
  if (!voc_loaded) {
    LOG(ERROR) << "Wrong path to vocabulary.";
    LOG(FATAL) << "Failed to open at: " << voc_file;
  }

  LOG(INFO) << "Vocabulary loaded!";

  // Create KeyFrame Database
  keyframe_database_ = new KeyFrameDatabase(*vocabulary_);

  // Create the Map
  map_ = new Map();

  // Create Drawers. These are used by the Viewer
  frame_drawer_ = new FrameDrawer(map_);
  map_drawer_ = new MapDrawer(map_, string_setting_file);

  // Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this
  // constructor)
  tracker_ = new Tracking(this, vocabulary_, frame_drawer_, map_drawer_, map_,
                          keyframe_database_, string_setting_file);

  // Initialize the Local Mapping thread and launch
  local_mapper_ = new LocalMapping(map_, true);
  thread_local_mapping_ =
      new thread(&ORB_SLAM2::LocalMapping::Run, local_mapper_);

  // Initialize the Loop Closing thread and launch
  loop_closer_ = new LoopClosing(map_, keyframe_database_, vocabulary_, false);
  thread_loop_closing_ = new thread(&ORB_SLAM2::LoopClosing::Run, loop_closer_);

  // Initialize the Viewer thread and launch
  viewer_ = new Viewer(this, frame_drawer_, map_drawer_, tracker_,
                       string_setting_file);
  thread_viewer_ = new thread(&Viewer::Run, viewer_);
  tracker_->SetViewer(viewer_);

  // Set pointers between threads
  tracker_->SetLocalMapper(local_mapper_);
  tracker_->SetLoopClosing(loop_closer_);

  local_mapper_->SetTracker(tracker_);
  local_mapper_->SetLoopCloser(loop_closer_);

  loop_closer_->SetTracker(tracker_);
  loop_closer_->SetLocalMapper(local_mapper_);
}

Eigen::Matrix4d MonoORBSlam::TrackMonocular(const cv::Mat& img,
                                            const double& timestamp) {
  {
    unique_lock<mutex> lock(mutex_mode_);
    if (is_activate_localization_mode_) {
      local_mapper_->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!local_mapper_->isStopped()) {
        usleep(1000);
      }
      tracker_->InformOnlyTracking(true);
      is_activate_localization_mode_ = false;
    }

    if (is_deactivate_localization_mode_) {
      tracker_->InformOnlyTracking(false);
      local_mapper_->Release();
      is_deactivate_localization_mode_ = false;
    }
  }

  {
    unique_lock<mutex> lock(mutex_reset_);
    if (do_reset_) {
      tracker_->Reset();
      do_reset_ = false;
    }
  }

  Eigen::Matrix4d Tcw = tracker_->GrabImageMonocular(img, timestamp);

  unique_lock<mutex> lock2(mutex_state_);
  tracking_state_ = tracker_->state_;
  tracked_map_points_ = tracker_->current_frame_.map_points_;
  tracked_undistort_keypoints_ = tracker_->current_frame_.undistort_keypoints_;

  return Tcw;
}

void MonoORBSlam::ActivateLocalizationMode() {
  unique_lock<mutex> lock(mutex_mode_);
  is_activate_localization_mode_ = true;
}

void MonoORBSlam::DeactivateLocalizationMode() {
  unique_lock<mutex> lock(mutex_mode_);
  is_deactivate_localization_mode_ = true;
}

bool MonoORBSlam::MapChanged() {
  static int n = 0;
  int curn = map_->GetLastBigChangeIdx();
  if (n < curn) {
    n = curn;
    return true;
  } else {
    return false;
  }
}

void MonoORBSlam::Reset() {
  unique_lock<mutex> lock(mutex_reset_);
  do_reset_ = true;
}

void MonoORBSlam::Shutdown() {
  local_mapper_->RequestFinish();
  loop_closer_->RequestFinish();
  if (viewer_) {
    viewer_->RequestFinish();
    while (!viewer_->isFinished()) {
      usleep(5000);
    }
  }
  // Wait until all thread have effectively stopped
  while (!local_mapper_->isFinished() || !loop_closer_->isFinished() ||
         loop_closer_->isRunningGBA()) {
    usleep(5000);
  }

  if (viewer_) {
    pangolin::BindToContext("ORB-SLAM2: Map Viewer");
  }
}

void MonoORBSlam::SaveTrajectoryTUM(const string& filename) {
  LOG(INFO) << "Cannot save camera trajectory for mono";
  return;
}

void MonoORBSlam::SaveMap(const string& filename) {
  LOG(INFO) << "Saving map to " << filename << "...";

  cv::FileStorage f(filename, cv::FileStorage::WRITE);

  LOG(INFO) << "Saving map points" << "...";
  f << "MapPoints" << "[";

  std::vector<MapPoint*> map_points = map_->GetAllMapPoints();
  sort(map_points.begin(), map_points.end(), MapPoint::lId);
  for (const auto& map_point : map_points) {
    if (map_point->isBad())
      continue;
    cv::Mat pos = MatEigenConverter::Vector3dToMat(map_point->GetWorldPos());
    f << "{:"
      << "id" << std::to_string(map_point->id_)
      << "pos" << pos
      << "descriptor" << map_point->GetDescriptor()
      << "}";
  }
  f << "]";
  LOG(INFO) << "map points saved!";

  LOG(INFO) << "Saving keyframes " << filename << " ...";
  std::vector<KeyFrame*> keyframes = map_->GetAllKeyFrames();
  sort(keyframes.begin(), keyframes.end(), KeyFrame::lId);
  f << "KeyFrames" << "[";
  for (const auto& keyframe : keyframes) {
    if (keyframe->isBad())
      continue;
    cv::Mat R =
        MatEigenConverter::Matrix3dToMat(keyframe->GetRotation().transpose());
    cv::Mat t = MatEigenConverter::Vector3dToMat(keyframe->GetCameraCenter());
    std::set<MapPoint*> map_point_set = keyframe->GetMapPoints();
    cv::Mat map_point_indices(1, map_point_set.size(), CV_32F);
    int count = 0;
    for (auto it = map_point_set.begin(); it != map_point_set.end(); it++) {
      map_point_indices.at<float>(0, count) = (*it)->id_;
      count++;
    }

    f << "{:"
      << "id" << std::to_string(keyframe->id_)
      << "timestamp" << keyframe->timestamp_
      << "R" << R
      << "t" << t
      << "map_point indices" << map_point_indices
      << "}";
  }
  f << "]";

  f.release();
  LOG(INFO) << "Map saved!";
}

void MonoORBSlam::SaveKeyFrameTrajectoryTUM(const string& filename) {
  LOG(INFO) << "Saving keyframe trajectory to " << filename << " ...";

  std::vector<KeyFrame*> keyframes = map_->GetAllKeyFrames();
  sort(keyframes.begin(), keyframes.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  // cv::Mat Two = vpKFs[0]->GetPoseInverse();
  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  for (size_t i = 0; i < keyframes.size(); i++) {
    KeyFrame* keyframe = keyframes[i];

    if (keyframe->isBad()) {
      continue;
    }

    Eigen::Matrix3d R = keyframe->GetRotation().transpose();
    Eigen::Quaterniond q(R);
    Eigen::Vector3d t = keyframe->GetCameraCenter();
    f << setprecision(6) << keyframe->timestamp_ << setprecision(7) << " "
      << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y()
      << " " << q.z() << " " << q.w() << endl;
  }
  f.close();
  LOG(INFO) << "trajectory saved!";
}

int MonoORBSlam::GetTrackingState() {
  unique_lock<mutex> lock(mutex_state_);
  return tracking_state_;
}

std::vector<MapPoint*> MonoORBSlam::GetTrackedMapPoints() {
  unique_lock<mutex> lock(mutex_state_);
  return tracked_map_points_;
}

std::vector<cv::KeyPoint> MonoORBSlam::GetTrackedKeyPointsUn() {
  unique_lock<mutex> lock(mutex_state_);
  return tracked_undistort_keypoints_;
}

}  // namespace ORB_SLAM2
