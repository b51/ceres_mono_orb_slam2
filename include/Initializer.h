/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Initializer.h
 *
 *          Created On: Thu 05 Sep 2019 10:03:56 AM CST
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

#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "Frame.h"

namespace ORB_SLAM2 {

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD
// CASE.
class Initializer {
  typedef std::pair<int, int> Match;

 public:
  // Fix the reference frame
  Initializer(const Frame& reference_frame, float sigma = 1.0,
              int iterations = 200);

  // Computes in parallel a fundamental matrix and a homography
  // Selects a model and tries to recover the motion and the structure from
  // motion
  bool Initialize(const Frame& current_frame, const vector<int>& matches,
                  Eigen::Matrix3d& R21, Eigen::Vector3d& t21,
                  vector<Eigen::Vector3d>& vP3D, vector<bool>& is_triangulated);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  void FindHomography(vector<bool>& vbMatchesInliers, float& score,
                      Eigen::Matrix3d& H21);
  void FindFundamental(vector<bool>& vbInliers, float& score,
                       Eigen::Matrix3d& F21);

  Eigen::Matrix3d ComputeH21(const vector<cv::Point2f>& vP1,
                             const vector<cv::Point2f>& vP2);
  Eigen::Matrix3d ComputeF21(const vector<cv::Point2f>& vP1,
                             const vector<cv::Point2f>& vP2);

  float CheckHomography(const Eigen::Matrix3d& H21, const Eigen::Matrix3d& H12,
                        vector<bool>& vbMatchesInliers, float sigma);

  float CheckFundamental(const Eigen::Matrix3d& F21,
                         vector<bool>& vbMatchesInliers, float sigma);

  bool ReconstructF(vector<bool>& vbMatchesInliers, Eigen::Matrix3d& F21,
                    Eigen::Matrix3d& K, Eigen::Matrix3d& R21,
                    Eigen::Vector3d& t21, vector<Eigen::Vector3d>& vP3D,
                    vector<bool>& is_triangulated, float minParallax,
                    int minTriangulated);

  bool ReconstructH(vector<bool>& vbMatchesInliers, Eigen::Matrix3d& H21,
                    Eigen::Matrix3d& K, Eigen::Matrix3d& R21,
                    Eigen::Vector3d& t21, vector<Eigen::Vector3d>& vP3D,
                    vector<bool>& is_triangulated, float minParallax,
                    int minTriangulated);

  void Triangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                   const Eigen::Matrix<double, 3, 4>& P1,
                   const Eigen::Matrix<double, 3, 4>& P2, Eigen::Vector3d& x3D);

  void Normalize(const vector<cv::KeyPoint>& vKeys,
                 vector<cv::Point2f>& vNormalizedPoints, Eigen::Matrix3d& T);

  int CheckRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
              const vector<cv::KeyPoint>& vKeys1,
              const vector<cv::KeyPoint>& vKeys2,
              const vector<Match>& vMatches12, vector<bool>& vbInliers,
              const Eigen::Matrix3d& K, vector<Eigen::Vector3d>& vP3D,
              float th2, vector<bool>& vbGood, float& parallax);

  void DecomposeE(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1,
                  Eigen::Matrix3d& R2, Eigen::Vector3d& t);

  // Keypoints from Reference Frame (Frame 1)
  std::vector<cv::KeyPoint> reference_keypoints_;

  // Keypoints from Current Frame (Frame 2)
  std::vector<cv::KeyPoint> current_keypoints_;

  // Current Matches from Reference to Current
  std::vector<Match> index_matches_;
  std::vector<bool> is_index_matched_;

  // Calibration
  Eigen::Matrix3d K_;

  // Standard Deviation and Variance
  float sigma_, sigma2_;

  // Ransac max iterations
  int max_iterations_;

  // Ransac sets
  vector<vector<size_t> > ransac_sets_;
};

}  // namespace ORB_SLAM2

#endif  // INITIALIZER_H_
