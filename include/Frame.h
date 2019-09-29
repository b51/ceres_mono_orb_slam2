/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Frame.h
 *
 *          Created On: Wed 04 Sep 2019 03:46:11 PM CST
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

#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Geometry>

#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "lib/DBoW2/DBoW2/BowVector.h"
#include "lib/DBoW2/DBoW2/FeatureVector.h"

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

namespace ORB_SLAM2 {

class MapPoint;
class KeyFrame;

class Frame {
 public:
  Frame();

  // Copy constructor.
  Frame(const Frame& frame);

  // Constructor for Monocular cameras.
  Frame(const cv::Mat& imgGray, const double& timestamp,
        ORBextractor* extractor, ORBVocabulary* voc, cv::Mat& K,
        cv::Mat& distCoef, const float& bf, const float& thDepth);

  // Extract ORB on the image. 0 for left image and 1 for right image.
  void ExtractORB(int flag, const cv::Mat& img);

  // Compute Bag of Words representation.
  void ComputeBoW();

  // Set the camera pose.
  void SetPose(Eigen::Matrix4d Tcw);

  // Computes rotation, translation and camera center matrices from the camera
  // pose.
  void UpdatePoseMatrices();

  // Returns the camera center.
  inline Eigen::Vector3d GetCameraCenter() { return Ow_; }

  // Returns inverse of rotation
  inline Eigen::Matrix3d GetRotationInverse() { return Rwc_; }

  // Check if a MapPoint is in the frustum of the camera
  // and fill variables of the MapPoint to be used by the tracking
  bool isInFrustum(MapPoint* map_point, float viewingCosLimit);

  // Compute the cell of a keypoint (return false if outside the grid)
  bool PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY);

  vector<size_t> GetFeaturesInArea(const float& x, const float& y,
                                   const float& r, const int minLevel = -1,
                                   const int maxLevel = -1) const;

  // Search a match for each keypoint in the left image to a keypoint in the
  // right image. If there is a match, depth is computed and the right
  // coordinate associated to the left keypoint is stored.
  void ComputeStereoMatches();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  // Vocabulary used for relocalization.
  ORBVocabulary* orb_vocabulary_;

  // Feature extractor. The right is used only in the stereo case.
  ORBextractor *orb_extractor_left_, *orb_extractor_right_;

  // Frame timestamp.
  double timestamp_;

  // Calibration matrix and OpenCV distortion parameters.
  cv::Mat K_;
  static float fx_;
  static float fy_;
  static float cx_;
  static float cy_;
  static float invfx_;
  static float invfy_;
  cv::Mat dist_coef_;

  // Stereo baseline multiplied by fx.
  float bf_;

  // Stereo baseline in meters.
  float mb_;

  // Threshold close/far points. Close points are inserted from 1 view.
  // Far points are inserted as in the monocular case from 2 views.
  float threshold_depth_;

  // Number of KeyPoints.
  int N_;

  // Vector of keypoints (original for visualization) and undistorted (actually
  // used by the system). In the stereo case, mvKeysUn is redundant as images
  // must be rectified. In the RGB-D case, RGB images can be distorted.
  std::vector<cv::KeyPoint> keypoints_, right_keypoints_;
  std::vector<cv::KeyPoint> undistort_keypoints_;

  // Corresponding stereo coordinate and depth for each keypoint.
  // "Monocular" keypoints have a negative value.
  std::vector<float> mvuRight;
  std::vector<float> depthes_;

  // Bag of Words Vector structures.
  DBoW2::BowVector bow_vector_;
  DBoW2::FeatureVector feature_vector_;

  // ORB descriptor, each row associated to a keypoint.
  cv::Mat descriptors_, right_descriptors_;

  // MapPoints associated to keypoints, NULL pointer if no association.
  std::vector<MapPoint*> map_points_;

  // Flag to identify outlier associations.
  std::vector<bool> is_outliers_;

  // Keypoints are assigned to cells in a grid to reduce matching complexity
  // when projecting MapPoints.
  static float grid_element_width_inv_;
  static float grid_element_height_inv_;
  std::vector<std::size_t> grid_[FRAME_GRID_COLS][FRAME_GRID_ROWS];

  // Camera pose.
  Eigen::Matrix4d Tcw_;

  // Current and Next Frame id.
  static long unsigned int next_id_;
  long unsigned int id_;

  // Reference Keyframe.
  KeyFrame* reference_keyframe_;

  // Scale pyramid info.
  int n_scale_levels_;
  float scale_factor_;
  float log_scale_factor_;
  vector<float> scale_factors_;
  vector<float> inv_scale_factors_;
  vector<float> level_sigma2s_;
  vector<float> inv_level_sigma2s_;

  // Undistorted Image Bounds (computed once).
  static float min_x_;
  static float max_x_;
  static float min_y_;
  static float max_y_;

  static bool do_initial_computations_;

 private:
  // Undistort keypoints given OpenCV distortion parameters.
  // Only for the RGB-D case. Stereo must be already rectified!
  // (called in the constructor).
  void UndistortKeyPoints();

  // Computes image bounds for the undistorted image (called in the
  // constructor).
  void ComputeImageBounds(const cv::Mat& imgLeft);

  // Assign keypoints to the grid for speed up feature matching (called in the
  // constructor).
  void AssignFeaturesToGrid();

  // Rotation, translation and camera center
  Eigen::Matrix3d Rcw_;
  Eigen::Vector3d tcw_;
  Eigen::Matrix3d Rwc_;
  Eigen::Vector3d Ow_;  //==mtwc
};

}  // namespace ORB_SLAM2
#endif  // FRAME_H
