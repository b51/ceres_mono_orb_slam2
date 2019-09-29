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

#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include <Eigen/Geometry>
#include <mutex>

#include "Frame.h"
#include "KeyFrameDatabase.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "lib/DBoW2/DBoW2/BowVector.h"
#include "lib/DBoW2/DBoW2/FeatureVector.h"

namespace ORB_SLAM2 {
class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame {
 public:
  KeyFrame(Frame& frame, Map* map, KeyFrameDatabase* keyframe_database);

  // Pose functions
  void SetPose(const Eigen::Matrix4d& Tcw);
  Eigen::Matrix4d GetPose();
  Eigen::Matrix4d GetPoseInverse();
  Eigen::Vector3d GetCameraCenter();
  Eigen::Matrix3d GetRotation();
  Eigen::Vector3d GetTranslation();

  // Bag of Words Representation
  void ComputeBoW();

  // Covisibility graph functions
  void AddConnection(KeyFrame* keyframe, const int& weight);
  void EraseConnection(KeyFrame* keyframe);
  void UpdateConnections();
  void UpdateBestCovisibles();
  std::set<KeyFrame*> GetConnectedKeyFrames();
  std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();
  std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int& N);
  std::vector<KeyFrame*> GetCovisiblesByWeight(const int& w);
  int GetWeight(KeyFrame* keyframe);

  // Spanning tree functions
  void AddChild(KeyFrame* keyframe);
  void EraseChild(KeyFrame* keyframe);
  void ChangeParent(KeyFrame* keyframe);
  std::set<KeyFrame*> GetChilds();
  KeyFrame* GetParent();
  bool hasChild(KeyFrame* keyframe);

  // Loop Edges
  void AddLoopEdge(KeyFrame* keyframe);
  std::set<KeyFrame*> GetLoopEdges();

  // MapPoint observation functions
  void AddMapPoint(MapPoint* map_point, const size_t& index);
  void EraseMapPointMatch(const size_t& index);
  void EraseMapPointMatch(MapPoint* map_point);
  void ReplaceMapPointMatch(const size_t& index, MapPoint* map_point);
  std::set<MapPoint*> GetMapPoints();
  std::vector<MapPoint*> GetMapPointMatches();
  int TrackedMapPoints(const int& minObs);
  MapPoint* GetMapPoint(const size_t& index);

  // KeyPoint functions
  std::vector<size_t> GetFeaturesInArea(const float& x, const float& y,
                                        const float& r) const;

  // Image
  bool IsInImage(const float& x, const float& y) const;

  // Enable/Disable bad flag changes
  void SetNotErase();
  void SetErase();

  // Set/check bad flag
  void SetBadFlag();
  bool isBad();

  // Compute Scene Depth (q=2 median). Used in monocular.
  float ComputeSceneMedianDepth(const int q);

  static bool weightComp(int a, int b) { return a > b; }

  static bool lId(KeyFrame* keyframe_1, KeyFrame* keyframe_2) {
    return keyframe_1->id_ < keyframe_2->id_;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // The following variables are accesed from only 1 thread or never change (no
  // mutex needed).
 public:
  static long unsigned int next_id_;
  long unsigned int id_;
  const long unsigned int frame_id_;

  const double timestamp_;

  // Grid (to speed up feature matching)
  const int grid_cols_;
  const int grid_rows_;
  const float grid_element_width_inv_;
  const float grid_element_height_inv_;

  // Variables used by the tracking
  long unsigned int track_reference_for_frame_;
  long unsigned int fuse_target_for_keyframe_;

  // Variables used by the local mapping
  long unsigned int n_BA_local_for_keyframe_;
  long unsigned int n_BA_fixed_for_keyframe_;

  // Variables used by the keyframe database
  long unsigned int n_loop_query_;
  int n_loop_words_;
  float loop_score_;
  long unsigned int reloc_query_;
  int n_reloc_words_;
  float reloc_score_;

  // Variables used by loop closing
  Eigen::Matrix4d global_BA_Tcw_;
  Eigen::Matrix4d global_BA_Bef_Tcw_;
  long unsigned int n_BA_global_for_keyframe_;

  // Calibration parameters
  const float fx_, fy_, cx_, cy_, invfx_, invfy_, bf_, mb_, threshold_depth_;

  // Number of KeyPoints
  const int N_;

  // KeyPoints, stereo coordinate and descriptors (all associated by an index)
  const std::vector<cv::KeyPoint> keypoints_;
  const std::vector<cv::KeyPoint> undistort_keypoints_;
  const std::vector<float> mvuRight;  // negative value for monocular points
  const std::vector<float> depthes_;  // negative value for monocular points
  const cv::Mat descriptors_;

  // BoW
  DBoW2::BowVector bow_vector_;
  DBoW2::FeatureVector feature_vector_;

  // Pose relative to parent (this is computed when bad flag is activated)
  Eigen::Matrix4d Tcp_;

  // Scale
  const int n_scale_levels_;
  const float scale_factor_;
  const float log_scale_factor_;
  const std::vector<float> scale_factors_;
  const std::vector<float> level_sigma2s_;
  const std::vector<float> inv_level_sigma2s_;

  // Image bounds and calibration
  const int min_x_;
  const int min_y_;
  const int max_x_;
  const int max_y_;
  const cv::Mat K_;

  // The following variables need to be accessed trough a mutex to be thread
  // safe.
 protected:
  // SE3 Pose and camera center
  Eigen::Matrix4d Tcw_;
  Eigen::Matrix4d Twc_;
  Eigen::Vector3d Ow_;

  // MapPoints associated to keypoints
  std::vector<MapPoint*> map_points_;

  // BoW
  KeyFrameDatabase* keyframe_database_;
  ORBVocabulary* orb_vocabulary_;

  // Grid over the image to speed up feature matching
  std::vector<std::vector<std::vector<size_t> > > grid_;

  std::map<KeyFrame*, int> connected_keyframe_weights_;
  std::vector<KeyFrame*> ordered_connected_keyframes_;
  std::vector<int> ordered_weights_;

  // Spanning Tree and Loop Edges
  bool is_first_connection_;
  KeyFrame* parent_;
  std::set<KeyFrame*> childrens_;
  std::set<KeyFrame*> loop_edges_;

  // Bad flags
  bool do_not_erase_;
  bool do_to_be_erased_;
  bool is_bad_;

  float half_baseline_;  // Only for visualization

  Map* map_;

  std::mutex mutex_pose_;
  std::mutex mutex_connections_;
  std::mutex mutex_features_;
};
}  // namespace ORB_SLAM2

#endif  // KEYFRAME_H_
