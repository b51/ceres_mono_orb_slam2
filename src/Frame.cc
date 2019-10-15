/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Frame.cc
 *
 *          Created On: Wed 04 Sep 2019 03:45:52 PM CST
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

#include "Frame.h"

#include <glog/logging.h>
#include <thread>

#include "MatEigenConverter.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2 {

long unsigned int Frame::next_id_ = 0;
bool Frame::do_initial_computations_ = true;
float Frame::cx_, Frame::cy_, Frame::fx_, Frame::fy_, Frame::invfx_,
    Frame::invfy_;
float Frame::min_x_, Frame::min_y_, Frame::max_x_, Frame::max_y_;
float Frame::grid_element_width_inv_, Frame::grid_element_height_inv_;

Frame::Frame(): Tcw_(Eigen::Matrix4d::Identity()) {}

// Copy Constructor
Frame::Frame(const Frame& frame)
    : orb_vocabulary_(frame.orb_vocabulary_),
      orb_extractor_left_(frame.orb_extractor_left_),
      orb_extractor_right_(frame.orb_extractor_right_),
      timestamp_(frame.timestamp_),
      K_(frame.K_.clone()),
      dist_coef_(frame.dist_coef_.clone()),
      bf_(frame.bf_),
      mb_(frame.mb_),
      threshold_depth_(frame.threshold_depth_),
      N_(frame.N_),
      keypoints_(frame.keypoints_),
      right_keypoints_(frame.right_keypoints_),
      undistort_keypoints_(frame.undistort_keypoints_),
      mvuRight(frame.mvuRight),
      depthes_(frame.depthes_),
      bow_vector_(frame.bow_vector_),
      feature_vector_(frame.feature_vector_),
      descriptors_(frame.descriptors_.clone()),
      right_descriptors_(frame.right_descriptors_.clone()),
      map_points_(frame.map_points_),
      is_outliers_(frame.is_outliers_),
      id_(frame.id_),
      reference_keyframe_(frame.reference_keyframe_),
      n_scale_levels_(frame.n_scale_levels_),
      scale_factor_(frame.scale_factor_),
      log_scale_factor_(frame.log_scale_factor_),
      scale_factors_(frame.scale_factors_),
      inv_scale_factors_(frame.inv_scale_factors_),
      level_sigma2s_(frame.level_sigma2s_),
      inv_level_sigma2s_(frame.inv_level_sigma2s_) {
  for (int i = 0; i < FRAME_GRID_COLS; i++) {
    for (int j = 0; j < FRAME_GRID_ROWS; j++) {
      grid_[i][j] = frame.grid_[i][j];
    }
  }

  if (!frame.Tcw_.isIdentity()) SetPose(frame.Tcw_);
}

Frame::Frame(const cv::Mat& imgGray, const double& timestamp,
             ORBextractor* extractor, ORBVocabulary* voc, cv::Mat& K,
             cv::Mat& distCoef, const float& bf, const float& thDepth)
    : orb_vocabulary_(voc),
      orb_extractor_left_(extractor),
      orb_extractor_right_(static_cast<ORBextractor*>(nullptr)),
      timestamp_(timestamp),
      K_(K.clone()),
      dist_coef_(distCoef.clone()),
      bf_(bf),
      threshold_depth_(thDepth),
      Tcw_(Eigen::Matrix4d::Identity()) {
  // Frame ID
  id_ = next_id_++;

  // Scale Level Info
  n_scale_levels_ = orb_extractor_left_->GetLevels();
  scale_factor_ = orb_extractor_left_->GetScaleFactor();
  log_scale_factor_ = log(scale_factor_);
  scale_factors_ = orb_extractor_left_->GetScaleFactors();
  inv_scale_factors_ = orb_extractor_left_->GetInverseScaleFactors();
  level_sigma2s_ = orb_extractor_left_->GetScaleSigmaSquares();
  inv_level_sigma2s_ = orb_extractor_left_->GetInverseScaleSigmaSquares();

  // ORB extraction
  ExtractORB(0, imgGray);

  N_ = keypoints_.size();

  if (keypoints_.empty()) {
    return;
  }

  UndistortKeyPoints();

  // Set no stereo information
  mvuRight = vector<float>(N_, -1);
  depthes_ = vector<float>(N_, -1);

  map_points_ = vector<MapPoint*>(N_, static_cast<MapPoint*>(nullptr));
  is_outliers_ = vector<bool>(N_, false);

  // This is done only for the first Frame (or after a change in the
  // calibration)
  if (do_initial_computations_) {
    ComputeImageBounds(imgGray);

    grid_element_width_inv_ = static_cast<float>(FRAME_GRID_COLS) /
                              static_cast<float>(max_x_ - min_x_);
    grid_element_height_inv_ = static_cast<float>(FRAME_GRID_ROWS) /
                               static_cast<float>(max_y_ - min_y_);

    fx_ = K.at<float>(0, 0);
    fy_ = K.at<float>(1, 1);
    cx_ = K.at<float>(0, 2);
    cy_ = K.at<float>(1, 2);
    invfx_ = 1.0f / fx_;
    invfy_ = 1.0f / fy_;

    do_initial_computations_ = false;
  }

  mb_ = bf_ / fx_;

  AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * N_ / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for (unsigned int i = 0; i < FRAME_GRID_COLS; i++) {
    for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
      grid_[i][j].reserve(nReserve);
    }
  }

  for (int i = 0; i < N_; i++) {
    const cv::KeyPoint& kp = undistort_keypoints_[i];

    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY))
      grid_[nGridPosX][nGridPosY].push_back(i);
  }
}

void Frame::ExtractORB(int flag, const cv::Mat& img) {
  (*orb_extractor_left_)(img, cv::Mat(), keypoints_, descriptors_);
}

void Frame::SetPose(Eigen::Matrix4d Tcw) {
  Tcw_ = Tcw;
  UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices() {
  Rcw_ = Tcw_.block<3, 3>(0, 0);
  Rwc_ = Rcw_.transpose();
  tcw_ = Tcw_.block<3, 1>(0, 3);
  Ow_ = -Rcw_.transpose() * tcw_;
}

bool Frame::isInFrustum(MapPoint* map_point, float viewingCosLimit) {
  map_point->is_track_in_view_ = false;

  // 3D in absolute coordinates
  Eigen::Vector3d P = map_point->GetWorldPos();

  // 3D in camera coordinates
  const Eigen::Vector3d Pc = Rcw_ * P + tcw_;
  const float& PcX = Pc[0];
  const float& PcY = Pc[1];
  const float& PcZ = Pc[2];

  // Check positive depth
  if (PcZ < 0.0f) return false;

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx_ * PcX * invz + cx_;
  const float v = fy_ * PcY * invz + cy_;

  if (u < min_x_ || u > max_x_) return false;
  if (v < min_y_ || v > max_y_) return false;

  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = map_point->GetMaxDistanceInvariance();
  const float minDistance = map_point->GetMinDistanceInvariance();
  const Eigen::Vector3d PO = P - Ow_;
  const float dist = PO.norm();

  if (dist < minDistance || dist > maxDistance) return false;

  // Check viewing angle
  Eigen::Vector3d Pn = map_point->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if (viewCos < viewingCosLimit) return false;

  // Predict scale in the image
  const int nPredictedLevel = map_point->PredictScale(dist, this);

  // Data used by the tracking
  map_point->is_track_in_view_ = true;
  map_point->track_proj_x_ = u;
  map_point->track_proj_x_r_ = u - bf_ * invz;
  map_point->track_proj_y_ = v;
  map_point->track_scale_level_ = nPredictedLevel;
  map_point->track_view_cos_ = viewCos;

  return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float& x, const float& y,
                                             const float& r, const int minLevel,
                                             const int maxLevel) const {
  std::vector<size_t> indices;
  indices.reserve(N_);

  const int min_cell_x =
      max(0, (int)floor((x - min_x_ - r) * grid_element_width_inv_));
  if (min_cell_x >= FRAME_GRID_COLS) {
    return indices;
  }

  const int max_cell_x =
      min((int)FRAME_GRID_COLS - 1,
          (int)ceil((x - min_x_ + r) * grid_element_width_inv_));
  if (max_cell_x < 0) {
    return indices;
  }

  const int min_cel_y =
      max(0, (int)floor((y - min_y_ - r) * grid_element_height_inv_));
  if (min_cel_y >= FRAME_GRID_ROWS) {
    return indices;
  }

  const int max_cell_y =
      min((int)FRAME_GRID_ROWS - 1,
          (int)ceil((y - min_y_ + r) * grid_element_height_inv_));
  if (max_cell_y < 0) {
    return indices;
  }

  const bool do_check_levels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = min_cell_x; ix <= max_cell_x; ix++) {
    for (int iy = min_cel_y; iy <= max_cell_y; iy++) {
      const vector<size_t> vCell = grid_[ix][iy];
      if (vCell.empty()) {
        continue;
      }

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const cv::KeyPoint& undistort_keypoint = undistort_keypoints_[vCell[j]];
        if (do_check_levels) {
          if (undistort_keypoint.octave < minLevel) {
            continue;
          }
          if (maxLevel >= 0) {
            if (undistort_keypoint.octave > maxLevel) {
              continue;
            }
          }
        }

        const float distx = undistort_keypoint.pt.x - x;
        const float disty = undistort_keypoint.pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r) {
          indices.push_back(vCell[j]);
        }
      }
    }
  }
  return indices;
}

bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) {
  posX = round((kp.pt.x - min_x_) * grid_element_width_inv_);
  posY = round((kp.pt.y - min_y_) * grid_element_height_inv_);

  // Keypoint's coordinates are undistorted, which could cause to go out of the
  // image
  if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 ||
      posY >= FRAME_GRID_ROWS)
    return false;

  return true;
}

void Frame::ComputeBoW() {
  if (bow_vector_.empty()) {
    vector<cv::Mat> vCurrentDesc = MatEigenConverter::toDescriptorVector(descriptors_);
    orb_vocabulary_->transform(vCurrentDesc, bow_vector_, feature_vector_, 4);
  }
}

void Frame::UndistortKeyPoints() {
  if (dist_coef_.at<float>(0) == 0.0) {
    undistort_keypoints_ = keypoints_;
    return;
  }

  // Fill matrix with points
  cv::Mat mat(N_, 2, CV_32F);
  for (int i = 0; i < N_; i++) {
    mat.at<float>(i, 0) = keypoints_[i].pt.x;
    mat.at<float>(i, 1) = keypoints_[i].pt.y;
  }

  // Undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, K_, dist_coef_, cv::Mat(), K_);
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  undistort_keypoints_.resize(N_);
  for (int i = 0; i < N_; i++) {
    cv::KeyPoint kp = keypoints_[i];
    kp.pt.x = mat.at<float>(i, 0);
    kp.pt.y = mat.at<float>(i, 1);
    undistort_keypoints_[i] = kp;
  }
}

void Frame::ComputeImageBounds(const cv::Mat& imgLeft) {
  if (dist_coef_.at<float>(0) != 0.0) {
    cv::Mat mat(4, 2, CV_32F);
    mat.at<float>(0, 0) = 0.0;
    mat.at<float>(0, 1) = 0.0;
    mat.at<float>(1, 0) = imgLeft.cols;
    mat.at<float>(1, 1) = 0.0;
    mat.at<float>(2, 0) = 0.0;
    mat.at<float>(2, 1) = imgLeft.rows;
    mat.at<float>(3, 0) = imgLeft.cols;
    mat.at<float>(3, 1) = imgLeft.rows;

    // Undistort corners
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K_, dist_coef_, cv::Mat(), K_);
    mat = mat.reshape(1);

    min_x_ = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
    max_x_ = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
    min_y_ = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
    max_y_ = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

  } else {
    min_x_ = 0.0f;
    max_x_ = imgLeft.cols;
    min_y_ = 0.0f;
    max_y_ = imgLeft.rows;
  }
}

void Frame::ComputeStereoMatches() {
  mvuRight = std::vector<float>(N_, -1.0f);
  depthes_ = std::vector<float>(N_, -1.0f);

  const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

  const int nRows = orb_extractor_left_->mvImagePyramid[0].rows;

  // Assign keypoints to row table
  std::vector<std::vector<size_t> > vRowIndices(nRows, vector<size_t>());

  for (int i = 0; i < nRows; i++) {
    vRowIndices[i].reserve(200);
  }

  const int Nr = right_keypoints_.size();

  for (int iR = 0; iR < Nr; iR++) {
    const cv::KeyPoint& kp = right_keypoints_[iR];
    const float& kpY = kp.pt.y;
    const float r = 2.0f * scale_factors_[right_keypoints_[iR].octave];
    const int maxr = ceil(kpY + r);
    const int minr = floor(kpY - r);

    for (int yi = minr; yi <= maxr; yi++) {
      vRowIndices[yi].push_back(iR);
    }
  }

  // Set limits for search
  const float minZ = mb_;
  const float minD = 0;
  const float maxD = bf_ / minZ;

  // For each left keypoint search a match in the right image
  vector<pair<int, int> > vDistIdx;
  vDistIdx.reserve(N_);

  for (int iL = 0; iL < N_; iL++) {
    const cv::KeyPoint& kpL = keypoints_[iL];
    const int& levelL = kpL.octave;
    const float& vL = kpL.pt.y;
    const float& uL = kpL.pt.x;

    const vector<size_t>& vCandidates = vRowIndices[vL];

    if (vCandidates.empty()) {
      continue;
    }

    const float minU = uL - maxD;
    const float maxU = uL - minD;

    if (maxU < 0) {
      continue;
    }

    int bestDist = ORBmatcher::TH_HIGH;
    size_t bestIdxR = 0;

    const cv::Mat& dL = descriptors_.row(iL);

    // Compare descriptor to right keypoints
    for (size_t iC = 0; iC < vCandidates.size(); iC++) {
      const size_t iR = vCandidates[iC];
      const cv::KeyPoint& kpR = right_keypoints_[iR];

      if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1) {
        continue;
      }

      const float& uR = kpR.pt.x;

      if (uR >= minU && uR <= maxU) {
        const cv::Mat& dR = right_descriptors_.row(iR);
        const int dist = ORBmatcher::DescriptorDistance(dL, dR);

        if (dist < bestDist) {
          bestDist = dist;
          bestIdxR = iR;
        }
      }
    }

    // Subpixel match by correlation
    if (bestDist < thOrbDist) {
      // coordinates in image pyramid at keypoint scale
      const float uR0 = right_keypoints_[bestIdxR].pt.x;
      const float scaleFactor = inv_scale_factors_[kpL.octave];
      const float scaleduL = round(kpL.pt.x * scaleFactor);
      const float scaledvL = round(kpL.pt.y * scaleFactor);
      const float scaleduR0 = round(uR0 * scaleFactor);

      // sliding window search
      const int w = 5;
      cv::Mat IL = orb_extractor_left_->mvImagePyramid[kpL.octave]
                       .rowRange(scaledvL - w, scaledvL + w + 1)
                       .colRange(scaleduL - w, scaleduL + w + 1);
      IL.convertTo(IL, CV_32F);
      IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

      int bestDist = INT_MAX;
      int bestincR = 0;
      const int L = 5;
      vector<float> vDists;
      vDists.resize(2 * L + 1);

      const float iniu = scaleduR0 + L - w;
      const float endu = scaleduR0 + L + w + 1;
      if (iniu < 0 ||
          endu >= orb_extractor_right_->mvImagePyramid[kpL.octave].cols) {
        continue;
      }

      for (int incR = -L; incR <= +L; incR++) {
        cv::Mat IR =
            orb_extractor_right_->mvImagePyramid[kpL.octave]
                .rowRange(scaledvL - w, scaledvL + w + 1)
                .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
        IR.convertTo(IR, CV_32F);
        IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

        float dist = cv::norm(IL, IR, cv::NORM_L1);
        if (dist < bestDist) {
          bestDist = dist;
          bestincR = incR;
        }

        vDists[L + incR] = dist;
      }

      if (bestincR == -L || bestincR == L) {
        continue;
      }

      // Sub-pixel match (Parabola fitting)
      const float dist1 = vDists[L + bestincR - 1];
      const float dist2 = vDists[L + bestincR];
      const float dist3 = vDists[L + bestincR + 1];

      const float deltaR =
          (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

      if (deltaR < -1 || deltaR > 1) {
        continue;
      }

      // Re-scaled coordinate
      float bestuR = scale_factors_[kpL.octave] *
                     ((float)scaleduR0 + (float)bestincR + deltaR);

      float disparity = (uL - bestuR);

      if (disparity >= minD && disparity < maxD) {
        if (disparity <= 0) {
          disparity = 0.01;
          bestuR = uL - 0.01;
        }
        depthes_[iL] = bf_ / disparity;
        mvuRight[iL] = bestuR;
        vDistIdx.push_back(pair<int, int>(bestDist, iL));
      }
    }
  }

  sort(vDistIdx.begin(), vDistIdx.end());
  const float median = vDistIdx[vDistIdx.size() / 2].first;
  const float thDist = 1.5f * 1.4f * median;

  for (int i = vDistIdx.size() - 1; i >= 0; i--) {
    if (vDistIdx[i].first < thDist) {
      break;
    } else {
      mvuRight[vDistIdx[i].second] = -1;
      depthes_[vDistIdx[i].second] = -1;
    }
  }
}

}  // namespace ORB_SLAM2
