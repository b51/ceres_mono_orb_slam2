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

#ifndef TRACKING_H_
#define TRACKING_H_

#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapDrawer.h"
#include "MonoORBSlam.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Viewer.h"

namespace ORB_SLAM2 {

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class MonoORBSlam;

class Tracking {
 public:
  Tracking(MonoORBSlam* mono_orb_slam, ORBVocabulary* vocabulary,
           FrameDrawer* frame_drawer, MapDrawer* map_drawer, Map* map,
           KeyFrameDatabase* keyframe_database,
           const string& string_setting_file);

  // Preprocess the input and call Track(). Extract features and performs
  Eigen::Matrix4d GrabImageMonocular(const cv::Mat& img, const double& timestamp);

  void SetLocalMapper(LocalMapping* local_mapper);
  void SetLoopClosing(LoopClosing* loop_closing);
  void SetViewer(Viewer* viewer);

  // Use this function if you have deactivated local mapping and you only want
  // to localize the camera.
  void InformOnlyTracking(const bool& flag);

 public:
  // Tracking states
  enum eTrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
  };

  eTrackingState state_;
  eTrackingState last_processed_state_;

  // Current Frame
  Frame current_frame_;
  cv::Mat img_gray_;

  // Initialization Variables (Monocular)
  std::vector<int> init_last_matches_;
  std::vector<int> init_matches_;
  std::vector<cv::Point2f> pre_matched_keypoints_;
  std::vector<Eigen::Vector3d> init_P3Ds_;
  Frame init_frame_;

  // Lists used to recover the full camera trajectory at the end of the
  // execution. Basically we store the reference keyframe for each frame and its
  // relative transformation
  list<Eigen::Matrix4d> relative_frame_poses_;
  list<KeyFrame*> reference_keyframes_;
  list<double> frame_times_;
  list<bool> do_lostes_;

  // True if local mapping is deactivated and we are performing only
  // localization
  bool do_only_tracking_;

  void Reset();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  bool Mapping();
  bool TrackingWithKnownMap();
  // Main tracking function. It is independent of the input sensor.
  void Track();

  // Map initialization for stereo and RGB-D
  void StereoInitialization();

  // Map initialization for monocular
  void MonocularInitialization();
  void CreateInitialMapMonocular();

  void CheckReplacedInLastFrame();
  bool TrackReferenceKeyFrame();
  void UpdateLastFrame();
  bool TrackWithMotionModel();

  bool Relocalization();

  void UpdateLocalMap();
  void UpdateLocalPoints();
  void UpdateLocalKeyFrames();

  bool TrackLocalMap();
  void SearchLocalPoints();

  bool NeedNewKeyFrame();
  void CreateNewKeyFrame();

  // In case of performing only localization, this flag is true when there are
  // no matches to points in the map. Still tracking will continue if there are
  // enough matches with temporal points. In that case we are doing visual
  // odometry. The system will try to do relocalization to recover "zero-drift"
  // localization to the map.
  bool do_vo_;

  // Other Thread Pointers
  LocalMapping* local_mapper_;
  LoopClosing* loop_closing_;

  // ORB
  ORBextractor *orb_extractor_left_, *orb_extractor_right_;
  ORBextractor* init_orb_extractor_;

  // BoW
  ORBVocabulary* orb_vocabulary_;
  KeyFrameDatabase* keyframe_database_;

  // Initalization (only for monocular)
  Initializer* initializer_;

  // Local Map
  KeyFrame* reference_keyframe_;
  std::vector<KeyFrame*> local_keyframes_;
  std::vector<MapPoint*> local_map_points_;

  // MonoORBSlam
  MonoORBSlam* mono_orb_slam_;

  // Drawers
  Viewer* viewer_;
  FrameDrawer* frame_drawer_;
  MapDrawer* map_drawer_;

  // Map
  Map* map_;

  // Calibration matrix
  cv::Mat K_;
  cv::Mat dist_coef_;
  float bf_;

  // New KeyFrame rules (according to fps)
  int min_frames_;
  int max_frames_;

  // Threshold close/far points
  // Points seen as close by the stereo/RGBD sensor are considered reliable
  // and inserted from just one frame. Far points requiere a match in two
  // keyframes.
  float theshold_depth_;

  // Current matches in frame
  int n_matches_inliers_;

  // Last Frame, KeyFrame and Relocalisation Info
  KeyFrame* last_keyframe_;
  Frame last_frame_;
  unsigned int last_keyframe_id_;
  unsigned int last_reloc_frame_id_;

  // Motion Model
  Eigen::Matrix4d velocity_;

  // Color order (true RGB, false BGR, ignored if grayscale)
  bool is_rgb_;
};

}  // namespace ORB_SLAM2

#endif  // TRACKING_H
