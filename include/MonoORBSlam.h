/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: MonoORBSlam.h
 *
 *          Created On: Thu 05 Sep 2019 12:05:02 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef MONO_ORB_SLAM_H_
#define MONO_ORB_SLAM_H_

#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <string>
#include <thread>
#include <glog/logging.h>

#include "FrameDrawer.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapDrawer.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
#include "Viewer.h"
#include "MatEigenConverter.h"

namespace ORB_SLAM2 {

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class MonoORBSlam {
 public:
  MonoORBSlam(const string& voc_file, const string& string_setting_file,
              const bool is_use_viewer = true);

  Eigen::Matrix4d TrackMonocular(const cv::Mat& img, const double& timestamp);

  // This stops local mapping thread (map building) and performs only camera
  // tracking.
  void ActivateLocalizationMode();
  // This resumes local mapping thread and performs SLAM again.
  void DeactivateLocalizationMode();

  // Returns true if there have been a big map change (loop closure, global BA)
  // since last call to this function
  bool MapChanged();

  // Reset the system (clear map)
  void Reset();

  // All threads will be requested to finish.
  // It waits until all threads have finished.
  // This function must be called before saving the trajectory.
  void Shutdown();

  // Save camera trajectory in the TUM RGB-D dataset format.
  // Only for stereo and RGB-D. This method does not work for monocular.
  // Call first Shutdown()
  // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
  void SaveTrajectoryTUM(const string& filename);

  // Save keyframe poses in the TUM RGB-D dataset format.
  // This method works for all sensor input.
  // Call first Shutdown()
  // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
  void SaveKeyFrameTrajectoryTUM(const string& filename);

  // TODO: Save/Load functions
  void SaveMap(const string &filename);
  // LoadMap(const string &filename);

  // Information from most recent processed frame
  // You can call this right after TrackMonocular (or stereo or RGBD)
  int GetTrackingState();
  std::vector<MapPoint*> GetTrackedMapPoints();
  std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

 private:
  // ORB vocabulary used for place recognition and feature matching.
  ORBVocabulary* vocabulary_;

  // KeyFrame database for place recognition (relocalization and loop
  // detection).
  KeyFrameDatabase* keyframe_database_;

  // Map structure that stores the pointers to all KeyFrames and MapPoints.
  Map* map_;

  // Tracker. It receives a frame and computes the associated camera pose.
  // It also decides when to insert a new keyframe, create some new MapPoints
  // and performs relocalization if tracking fails.
  Tracking* tracker_;

  // Local Mapper. It manages the local map and performs local bundle
  // adjustment.
  LocalMapping* local_mapper_;

  // Loop Closer. It searches loops with every new keyframe. If there is a
  // loop it performs a pose graph optimization and full bundle adjustment
  // (in a new thread) afterwards.
  LoopClosing* loop_closer_;

  // The viewer draws the map and the current camera pose. It uses Pangolin.
  Viewer* viewer_;

  FrameDrawer* frame_drawer_;
  MapDrawer* map_drawer_;

  // System threads: Local Mapping, Loop Closing, Viewer.
  // The Tracking thread "lives" in the main execution thread that creates the
  // System object.
  std::thread* thread_local_mapping_;
  std::thread* thread_loop_closing_;
  std::thread* thread_viewer_;

  // Reset flag
  std::mutex mutex_reset_;
  bool do_reset_;

  // Change mode flags
  std::mutex mutex_mode_;
  bool is_activate_localization_mode_;
  bool is_deactivate_localization_mode_;

  // Tracking state
  int tracking_state_;
  std::vector<MapPoint*> tracked_map_points_;
  std::vector<cv::KeyPoint> tracked_undistort_keypoints_;
  std::mutex mutex_state_;
};

}  // namespace ORB_SLAM2

#endif
