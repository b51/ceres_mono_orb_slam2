/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Sim3Solver.cc
 *
 *          Created On: Fri 20 Sep 2019 05:49:39 PM CST
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

#include "Sim3Solver.h"

#include <cmath>
#include <opencv2/core/core.hpp>
#include <vector>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "lib/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2 {

Sim3Solver::Sim3Solver(KeyFrame* keyframe_1, KeyFrame* keyframe_2,
                       const vector<MapPoint*>& matched_points_1_in_2,
                       const bool is_fixed_scale)
    : n_iterations_(0), n_best_inliers_(0), is_fixed_scale_(is_fixed_scale) {
  keyframe_1_ = keyframe_1;
  keyframe_2_ = keyframe_2;

  vector<MapPoint*> matched_points_in_1 = keyframe_1->GetMapPointMatches();

  N1_ = matched_points_1_in_2.size();

  map_points_1_.reserve(N1_);
  map_points_2_.reserve(N1_);
  matched_points_1_in_2_ = matched_points_1_in_2;
  matched_indices_1_.reserve(N1_);
  X3Dsc1_.reserve(N1_);
  X3Dsc2_.reserve(N1_);

  cv::Mat Rcw1 = keyframe_1->GetRotation();
  cv::Mat tcw1 = keyframe_1->GetTranslation();
  cv::Mat Rcw2 = keyframe_2->GetRotation();
  cv::Mat tcw2 = keyframe_2->GetTranslation();

  all_indices_.reserve(N1_);

  size_t idx = 0;
  for (int i1 = 0; i1 < N1_; i1++) {
    if (matched_points_1_in_2[i1]) {
      MapPoint* map_point_1 = matched_points_in_1[i1];
      MapPoint* map_point_2 = matched_points_1_in_2[i1];

      if (!map_point_1) continue;

      if (map_point_1->isBad() || map_point_2->isBad()) continue;

      int indexKF1 = map_point_1->GetIndexInKeyFrame(keyframe_1);
      int indexKF2 = map_point_2->GetIndexInKeyFrame(keyframe_2);

      if (indexKF1 < 0 || indexKF2 < 0) continue;

      const cv::KeyPoint& kp1 = keyframe_1->undistort_keypoints_[indexKF1];
      const cv::KeyPoint& kp2 = keyframe_2->undistort_keypoints_[indexKF2];

      const float sigmaSquare1 = keyframe_1->level_sigma2s_[kp1.octave];
      const float sigmaSquare2 = keyframe_2->level_sigma2s_[kp2.octave];

      max_errors_1_.push_back(9.210 * sigmaSquare1);
      max_errors_2_.push_back(9.210 * sigmaSquare2);

      map_points_1_.push_back(map_point_1);
      map_points_2_.push_back(map_point_2);
      matched_indices_1_.push_back(i1);

      cv::Mat X3D1w = map_point_1->GetWorldPos();
      X3Dsc1_.push_back(Rcw1 * X3D1w + tcw1);

      cv::Mat X3D2w = map_point_2->GetWorldPos();
      X3Dsc2_.push_back(Rcw2 * X3D2w + tcw2);

      all_indices_.push_back(idx);
      idx++;
    }
  }

  K1_ = keyframe_1->K_;
  K2_ = keyframe_2->K_;

  FromCameraToImage(X3Dsc1_, points_1_in_im_1_, K1_);
  FromCameraToImage(X3Dsc2_, points_2_in_im_2_, K2_);

  SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int min_inliers,
                                     int max_iterations) {
  ransac_probability_ = probability;
  ransac_min_inliers_ = min_inliers;
  ransac_max_iterations_ = max_iterations;

  N_ = map_points_1_.size();  // number of correspondences

  is_inliers_i_.resize(N_);

  // Adjust Parameters according to number of correspondences
  float epsilon = (float)ransac_min_inliers_ / N_;

  // Set RANSAC iterations according to probability, epsilon, and max iterations
  int n_iterations;

  if (ransac_min_inliers_ == N_)
    n_iterations = 1;
  else
    n_iterations =
        ceil(log(1 - ransac_probability_) / log(1 - pow(epsilon, 3)));

  ransac_max_iterations_ = max(1, min(n_iterations, ransac_max_iterations_));

  n_iterations_ = 0;
}

cv::Mat Sim3Solver::iterate(int n_iterations, bool& is_no_more,
                            vector<bool>& is_inliers, int& n_inliers) {
  is_no_more = false;
  is_inliers = vector<bool>(N1_, false);
  n_inliers = 0;

  if (N_ < ransac_min_inliers_) {
    is_no_more = true;
    return cv::Mat();
  }

  vector<size_t> available_indices;

  cv::Mat P3Dc1i(3, 3, CV_32F);
  cv::Mat P3Dc2i(3, 3, CV_32F);

  int n_current_iterations = 0;
  while (n_iterations_ < ransac_max_iterations_ &&
         n_current_iterations < n_iterations) {
    n_current_iterations++;
    n_iterations_++;

    available_indices = all_indices_;

    // Get min set of points
    for (short i = 0; i < 3; ++i) {
      int randi = DUtils::Random::RandomInt(0, available_indices.size() - 1);

      int idx = available_indices[randi];

      X3Dsc1_[idx].copyTo(P3Dc1i.col(i));
      X3Dsc2_[idx].copyTo(P3Dc2i.col(i));

      available_indices[randi] = available_indices.back();
      available_indices.pop_back();
    }

    ComputeSim3(P3Dc1i, P3Dc2i);

    CheckInliers();

    if (n_inliers_i_ >= n_best_inliers_) {
      is_best_inliers_ = is_inliers_i_;
      n_best_inliers_ = n_inliers_i_;
      best_T12_ = T12i_.clone();
      best_rotation_ = R12i_.clone();
      best_translation_ = t12i_.clone();
      best_scale_ = scale_12_i_;

      if (n_inliers_i_ > ransac_min_inliers_) {
        n_inliers = n_inliers_i_;
        for (int i = 0; i < N_; i++)
          if (is_inliers_i_[i]) is_inliers[matched_indices_1_[i]] = true;
        return best_T12_;
      }
    }
  }

  if (n_iterations_ >= ransac_max_iterations_) is_no_more = true;

  return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool>& vbInliers12, int& n_inliers) {
  bool flag;
  return iterate(ransac_max_iterations_, flag, vbInliers12, n_inliers);
}

void Sim3Solver::ComputeCentroid(cv::Mat& P, cv::Mat& Pr, cv::Mat& C) {
  cv::reduce(P, C, 1, CV_REDUCE_SUM);
  C = C / P.cols;

  for (int i = 0; i < P.cols; i++) {
    Pr.col(i) = P.col(i) - C;
  }
}

void Sim3Solver::ComputeSim3(cv::Mat& P1, cv::Mat& P2) {
  // Custom implementation of:
  // Horn 1987, Closed-form solution of absolute orientataion using unit
  // quaternions

  // Step 1: Centroid and relative coordinates

  cv::Mat Pr1(P1.size(),
              P1.type());  // Relative coordinates to centroid (set 1)
  cv::Mat Pr2(P2.size(),
              P2.type());        // Relative coordinates to centroid (set 2)
  cv::Mat O1(3, 1, Pr1.type());  // Centroid of P1
  cv::Mat O2(3, 1, Pr2.type());  // Centroid of P2

  ComputeCentroid(P1, Pr1, O1);
  ComputeCentroid(P2, Pr2, O2);

  // Step 2: Compute M matrix

  cv::Mat M = Pr2 * Pr1.t();

  // Step 3: Compute N matrix

  double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

  cv::Mat N(4, 4, P1.type());

  N11 = M.at<float>(0, 0) + M.at<float>(1, 1) + M.at<float>(2, 2);
  N12 = M.at<float>(1, 2) - M.at<float>(2, 1);
  N13 = M.at<float>(2, 0) - M.at<float>(0, 2);
  N14 = M.at<float>(0, 1) - M.at<float>(1, 0);
  N22 = M.at<float>(0, 0) - M.at<float>(1, 1) - M.at<float>(2, 2);
  N23 = M.at<float>(0, 1) + M.at<float>(1, 0);
  N24 = M.at<float>(2, 0) + M.at<float>(0, 2);
  N33 = -M.at<float>(0, 0) + M.at<float>(1, 1) - M.at<float>(2, 2);
  N34 = M.at<float>(1, 2) + M.at<float>(2, 1);
  N44 = -M.at<float>(0, 0) - M.at<float>(1, 1) + M.at<float>(2, 2);

  N = (cv::Mat_<float>(4, 4) << N11, N12, N13, N14, N12, N22, N23, N24, N13,
       N23, N33, N34, N14, N24, N34, N44);

  // Step 4: Eigenvector of the highest eigenvalue

  cv::Mat eval, evec;

  cv::eigen(N, eval, evec);  // evec[0] is the quaternion of the desired
                             // rotation

  cv::Mat vec(1, 3, evec.type());
  (evec.row(0).colRange(1, 4))
      .copyTo(vec);  // extract imaginary part of the quaternion (sin*axis)

  // Rotation angle. sin is the norm of the imaginary part, cos is the real part
  double ang = atan2(norm(vec), evec.at<float>(0, 0));

  vec = 2 * ang * vec /
        norm(vec);  // Angle-axis representation. quaternion angle is the half

  R12i_.create(3, 3, P1.type());

  cv::Rodrigues(vec, R12i_);  // computes the rotation matrix from angle-axis

  // Step 5: Rotate set 2

  cv::Mat P3 = R12i_ * Pr2;

  // Step 6: Scale

  if (!is_fixed_scale_) {
    double nom = Pr1.dot(P3);
    cv::Mat aux_P3(P3.size(), P3.type());
    aux_P3 = P3;
    cv::pow(P3, 2, aux_P3);
    double den = 0;

    for (int i = 0; i < aux_P3.rows; i++) {
      for (int j = 0; j < aux_P3.cols; j++) {
        den += aux_P3.at<float>(i, j);
      }
    }

    scale_12_i_ = nom / den;
  } else
    scale_12_i_ = 1.0f;

  // Step 7: Translation

  t12i_.create(1, 3, P1.type());
  t12i_ = O1 - scale_12_i_ * R12i_ * O2;

  // Step 8: Transformation

  // Step 8.1 T12
  T12i_ = cv::Mat::eye(4, 4, P1.type());

  cv::Mat sR = scale_12_i_ * R12i_;

  sR.copyTo(T12i_.rowRange(0, 3).colRange(0, 3));
  t12i_.copyTo(T12i_.rowRange(0, 3).col(3));

  // Step 8.2 T21

  T21i_ = cv::Mat::eye(4, 4, P1.type());

  cv::Mat sRinv = (1.0 / scale_12_i_) * R12i_.t();

  sRinv.copyTo(T21i_.rowRange(0, 3).colRange(0, 3));
  cv::Mat tinv = -sRinv * t12i_;
  tinv.copyTo(T21i_.rowRange(0, 3).col(3));
}

void Sim3Solver::CheckInliers() {
  vector<cv::Mat> points_1_in_im_2, points_2_in_im_1;
  Project(X3Dsc2_, points_2_in_im_1, T12i_, K1_);
  Project(X3Dsc1_, points_1_in_im_2, T21i_, K2_);

  n_inliers_i_ = 0;

  for (size_t i = 0; i < points_1_in_im_1_.size(); i++) {
    cv::Mat dist1 = points_1_in_im_1_[i] - points_2_in_im_1[i];
    cv::Mat dist2 = points_1_in_im_2[i] - points_2_in_im_2_[i];

    const float err1 = dist1.dot(dist1);
    const float err2 = dist2.dot(dist2);

    if (err1 < max_errors_1_[i] && err2 < max_errors_2_[i]) {
      is_inliers_i_[i] = true;
      n_inliers_i_++;
    } else
      is_inliers_i_[i] = false;
  }
}

cv::Mat Sim3Solver::GetEstimatedRotation() { return best_rotation_.clone(); }

cv::Mat Sim3Solver::GetEstimatedTranslation() {
  return best_translation_.clone();
}

float Sim3Solver::GetEstimatedScale() { return best_scale_; }

void Sim3Solver::Project(const vector<cv::Mat>& vP3Dw, vector<cv::Mat>& vP2D,
                         cv::Mat Tcw, cv::Mat K) {
  cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
  cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
  const float& fx = K.at<float>(0, 0);
  const float& fy = K.at<float>(1, 1);
  const float& cx = K.at<float>(0, 2);
  const float& cy = K.at<float>(1, 2);

  vP2D.clear();
  vP2D.reserve(vP3Dw.size());

  for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++) {
    cv::Mat P3Dc = Rcw * vP3Dw[i] + tcw;
    const float invz = 1 / (P3Dc.at<float>(2));
    const float x = P3Dc.at<float>(0) * invz;
    const float y = P3Dc.at<float>(1) * invz;

    vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
  }
}

void Sim3Solver::FromCameraToImage(const vector<cv::Mat>& vP3Dc,
                                   vector<cv::Mat>& vP2D, cv::Mat K) {
  const float& fx = K.at<float>(0, 0);
  const float& fy = K.at<float>(1, 1);
  const float& cx = K.at<float>(0, 2);
  const float& cy = K.at<float>(1, 2);

  vP2D.clear();
  vP2D.reserve(vP3Dc.size());

  for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++) {
    const float invz = 1 / (vP3Dc[i].at<float>(2));
    const float x = vP3Dc[i].at<float>(0) * invz;
    const float y = vP3Dc[i].at<float>(1) * invz;

    vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
  }
}

}  // namespace ORB_SLAM2
