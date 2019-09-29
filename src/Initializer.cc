/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Initializer.cc
 *
 *          Created On: Thu 05 Sep 2019 10:03:29 AM CST
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

#include "Initializer.h"

#include "lib/DBoW2/DUtils/Random.h"

#include "MatEigenConverter.h"
#include "ORBmatcher.h"

#include <glog/logging.h>
#include <thread>

namespace ORB_SLAM2 {

Initializer::Initializer(const Frame& reference_frame, float sigma,
                         int iterations) {
  K_ = MatEigenConverter::MatToMatrix3d(reference_frame.K_);

  reference_keypoints_ = reference_frame.undistort_keypoints_;

  sigma_ = sigma;
  sigma2_ = sigma * sigma;
  max_iterations_ = iterations;
}

bool Initializer::Initialize(const Frame& current_frame,
                             const vector<int>& matches, Eigen::Matrix3d& R21,
                             Eigen::Vector3d& t21,
                             std::vector<Eigen::Vector3d>& vP3D,
                             vector<bool>& is_triangulated) {
  // Fill structures with current keypoints and matches with reference frame
  // Reference Frame: 1, Current Frame: 2
  current_keypoints_ = current_frame.undistort_keypoints_;

  index_matches_.clear();
  index_matches_.reserve(current_keypoints_.size());
  is_index_matched_.resize(reference_keypoints_.size());
  for (size_t i = 0, iend = matches.size(); i < iend; i++) {
    if (matches[i] >= 0) {
      index_matches_.push_back(make_pair(i, matches[i]));
      is_index_matched_[i] = true;
    } else
      is_index_matched_[i] = false;
  }

  const int N = index_matches_.size();

  // Indices for minimum set selection
  vector<size_t> vAllIndices;
  vAllIndices.reserve(N);
  vector<size_t> vAvailableIndices;

  for (int i = 0; i < N; i++) {
    vAllIndices.push_back(i);
  }

  // Generate sets of 8 points for each RANSAC iteration
  ransac_sets_ = vector<vector<size_t>>(max_iterations_, vector<size_t>(8, 0));

  DUtils::Random::SeedRandOnce(0);

  for (int it = 0; it < max_iterations_; it++) {
    vAvailableIndices = vAllIndices;

    // Select a minimum set
    for (size_t j = 0; j < 8; j++) {
      int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
      int index = vAvailableIndices[randi];

      ransac_sets_[it][j] = index;

      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();
    }
  }

  // Launch threads to compute in parallel a fundamental matrix and a homography
  vector<bool> vbMatchesInliersH, vbMatchesInliersF;
  float SH, SF;
  Eigen::Matrix3d H, F;

  thread threadH(&Initializer::FindHomography, this, ref(vbMatchesInliersH),
                 ref(SH), ref(H));
  thread threadF(&Initializer::FindFundamental, this, ref(vbMatchesInliersF),
                 ref(SF), ref(F));

  // Wait until both threads have finished
  threadH.join();
  threadF.join();

  // Compute ratio of scores
  float RH = SH / (SH + SF);

  // Try to reconstruct from homography or fundamental depending on the ratio
  // (0.40-0.45)
  if (RH > 0.40) {
    return ReconstructH(vbMatchesInliersH, H, K_, R21, t21, vP3D,
                        is_triangulated, 1.0, 50);
  } else {  // if(pF_HF>0.6)
    return ReconstructF(vbMatchesInliersF, F, K_, R21, t21, vP3D,
                        is_triangulated, 1.0, 50);
  }

  return false;
}

void Initializer::FindHomography(vector<bool>& vbMatchesInliers, float& score,
                                 Eigen::Matrix3d& H21) {
  // Number of putative matches
  const int N = index_matches_.size();

  // Normalize coordinates
  std::vector<cv::Point2f> vPn1, vPn2;
  Eigen::Matrix3d T1, T2;
  Normalize(reference_keypoints_, vPn1, T1);
  Normalize(current_keypoints_, vPn2, T2);

  // Best Results variables
  score = 0.0;
  vbMatchesInliers = vector<bool>(N, false);

  // Iteration variables
  vector<cv::Point2f> vPn1i(8);
  vector<cv::Point2f> vPn2i(8);
  Eigen::Matrix3d H21i, H12i;
  vector<bool> vbCurrentInliers(N, false);
  float currentScore;

  // Perform all RANSAC iterations and save the solution with highest score
  for (int it = 0; it < max_iterations_; it++) {
    // Select a minimum set
    for (size_t j = 0; j < 8; j++) {
      int index = ransac_sets_[it][j];

      vPn1i[j] = vPn1[index_matches_[index].first];
      vPn2i[j] = vPn2[index_matches_[index].second];
    }

    Eigen::Matrix3d Hn = ComputeH21(vPn1i, vPn2i);
    H21i = T2.inverse() * Hn * T1;
    H12i = H21i.inverse();

    currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, sigma_);

    if (currentScore > score) {
      H21 = H21i;
      vbMatchesInliers = vbCurrentInliers;
      score = currentScore;
    }
  }
}

void Initializer::FindFundamental(vector<bool>& vbMatchesInliers, float& score,
                                  Eigen::Matrix3d& F21) {
  // Number of putative matches
  const int N = vbMatchesInliers.size();

  // Normalize coordinates
  vector<cv::Point2f> vPn1, vPn2;
  Eigen::Matrix3d T1, T2;
  Normalize(reference_keypoints_, vPn1, T1);
  Normalize(current_keypoints_, vPn2, T2);

  // Best Results variables
  score = 0.0;
  vbMatchesInliers = vector<bool>(N, false);

  // Iteration variables
  vector<cv::Point2f> vPn1i(8);
  vector<cv::Point2f> vPn2i(8);
  Eigen::Matrix3d F21i;
  vector<bool> vbCurrentInliers(N, false);
  float currentScore;

  // Perform all RANSAC iterations and save the solution with highest score
  for (int it = 0; it < max_iterations_; it++) {
    // Select a minimum set
    for (int j = 0; j < 8; j++) {
      int index = ransac_sets_[it][j];

      vPn1i[j] = vPn1[index_matches_[index].first];
      vPn2i[j] = vPn2[index_matches_[index].second];
    }

    Eigen::Matrix3d Fn = ComputeF21(vPn1i, vPn2i);

    F21i = T2.transpose() * Fn * T1;

    currentScore = CheckFundamental(F21i, vbCurrentInliers, sigma_);

    if (currentScore > score) {
      F21 = F21i;
      vbMatchesInliers = vbCurrentInliers;
      score = currentScore;
    }
  }
}

// (u2, v2, w2).transpose = H21 * (u1, v1, w1).transpose
Eigen::Matrix3d Initializer::ComputeH21(const vector<cv::Point2f>& vP1,
                                        const vector<cv::Point2f>& vP2) {
  const int N = vP1.size();

  Eigen::MatrixXd A(2 * N, 9);
  A.setZero();
  for (int i = 0; i < N; i++) {
    const float u1 = vP1[i].x;
    const float v1 = vP1[i].y;
    const float u2 = vP2[i].x;
    const float v2 = vP2[i].y;

    A(2 * i, 0) = -u1;
    A(2 * i, 1) = -v1;
    A(2 * i, 2) = -1.;
    A(2 * i, 6) = u1 * u2;
    A(2 * i, 7) = v1 * u2;
    A(2 * i, 8) = u2;

    A(2 * i + 1, 3) = -u1;
    A(2 * i + 1, 4) = -v1;
    A(2 * i + 1, 5) = -1.;
    A(2 * i + 1, 6) = u1 * v2;
    A(2 * i + 1, 7) = v1 * v2;
    A(2 * i + 1, 8) = v2;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 9, 9> V = svd.matrixV();
  Eigen::Matrix3d H21 =
      Eigen::Map<Eigen::Matrix3d>(V.col(8).data()).transpose();

  return H21;
}

Eigen::Matrix3d Initializer::ComputeF21(const vector<cv::Point2f>& vP1,
                                        const vector<cv::Point2f>& vP2) {
  const int N = vP1.size();

  Eigen::MatrixXd A(N, 9);
  A.setZero();

  for (int i = 0; i < N; i++) {
    const float u1 = vP1[i].x;
    const float v1 = vP1[i].y;
    const float u2 = vP2[i].x;
    const float v2 = vP2[i].y;

    A(i, 0) = u2 * u1;
    A(i, 1) = u2 * v1;
    A(i, 2) = u2;
    A(i, 3) = v2 * u1;
    A(i, 4) = v2 * v1;
    A(i, 5) = v2;
    A(i, 6) = u1;
    A(i, 7) = v1;
    A(i, 8) = 1;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd_pre(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 9, 9> Vpre = svd_pre.matrixV();
  Eigen::Matrix3d Fpre =
      Eigen::Map<Eigen::Matrix3d>(Vpre.col(8).data()).transpose();

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      Fpre, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Vector3d singular = svd.singularValues();
  Eigen::DiagonalMatrix<double, 3> optimized_singular;
  optimized_singular.diagonal() << singular(0), singular(1), 0;

  Eigen::Matrix3d F21 = U * optimized_singular * V.transpose();

  return F21;
}

float Initializer::CheckHomography(const Eigen::Matrix3d& H21,
                                   const Eigen::Matrix3d& H12,
                                   vector<bool>& vbMatchesInliers,
                                   float sigma) {
  const int N = index_matches_.size();

  vbMatchesInliers.resize(N);

  float score = 0;

  const float th = 5.991;

  const float invSigmaSquare = 1.0 / (sigma * sigma);

  for (int i = 0; i < N; i++) {
    bool bIn = true;

    const cv::KeyPoint& kp1 = reference_keypoints_[index_matches_[i].first];
    const cv::KeyPoint& kp2 = current_keypoints_[index_matches_[i].second];

    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;

    Eigen::Vector3d x1(u1, v1, 1.);
    Eigen::Vector3d x2(u2, v2, 1.);

    // Reprojection error in first image
    // x2in1 = H12*x2
    Eigen::Vector3d x2in1 = H12 * x2;

    const float u2in1 = x2in1(0) / x2in1(2);
    const float v2in1 = x2in1(1) / x2in1(2);

    const float squareDist1 =
        (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

    const float chiSquare1 = squareDist1 * invSigmaSquare;

    if (chiSquare1 > th)
      bIn = false;
    else
      score += th - chiSquare1;

    // Reprojection error in second image
    // x1in2 = H21*x1

    Eigen::Vector3d x1in2 = H21 * x1;

    const float u1in2 = x1in2(0) / x1in2(2);
    const float v1in2 = x1in2(1) / x1in2(2);

    const float squareDist2 =
        (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

    const float chiSquare2 = squareDist2 * invSigmaSquare;

    if (chiSquare2 > th)
      bIn = false;
    else
      score += th - chiSquare2;

    if (bIn)
      vbMatchesInliers[i] = true;
    else
      vbMatchesInliers[i] = false;
  }

  return score;
}

float Initializer::CheckFundamental(const Eigen::Matrix3d& F21,
                                    vector<bool>& vbMatchesInliers,
                                    float sigma) {
  const int N = index_matches_.size();

  vbMatchesInliers.resize(N);

  float score = 0;

  const float th = 3.841;
  const float thScore = 5.991;

  const float invSigmaSquare = 1.0 / (sigma * sigma);

  for (int i = 0; i < N; i++) {
    bool bIn = true;

    const cv::KeyPoint& kp1 = reference_keypoints_[index_matches_[i].first];
    const cv::KeyPoint& kp2 = current_keypoints_[index_matches_[i].second];

    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;
    Eigen::Vector3d x1(u1, v1, 1.);
    Eigen::Vector3d x2(u2, v2, 1.);

    // Reprojection error in second image
    // l2=F21x1=(a2,b2,c2)
    Eigen::Vector3d l2 = F21 * x1;

    const float num2 = x2.transpose() * l2;

    const float squareDist1 = num2 * num2 / (l2(0) * l2(0) + l2(1) * l2(1));

    const float chiSquare1 = squareDist1 * invSigmaSquare;

    if (chiSquare1 > th)
      bIn = false;
    else
      score += thScore - chiSquare1;

    // Reprojection error in second image
    // l1 =x2tF21=(a1,b1,c1)
    // F21 Homogeneous matrix

    Eigen::Matrix<double, 1, 3> l1 = x2.transpose() * F21;

    const float num1 = l1 * x1;

    const float squareDist2 = num1 * num1 / (l1(0) * l1(0) + l1(1) * l1(1));

    const float chiSquare2 = squareDist2 * invSigmaSquare;

    if (chiSquare2 > th)
      bIn = false;
    else
      score += thScore - chiSquare2;

    if (bIn)
      vbMatchesInliers[i] = true;
    else
      vbMatchesInliers[i] = false;
  }

  return score;
}

bool Initializer::ReconstructF(vector<bool>& vbMatchesInliers,
                               Eigen::Matrix3d& F21, Eigen::Matrix3d& K,
                               Eigen::Matrix3d& R21, Eigen::Vector3d& t21,
                               std::vector<Eigen::Vector3d>& vP3D,
                               vector<bool>& is_triangulated, float minParallax,
                               int minTriangulated) {
  int N = 0;
  for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
    if (vbMatchesInliers[i]) N++;

  // Compute Essential Matrix from Fundamental Matrix
  Eigen::Matrix3d E21 = K.transpose() * F21 * K;

  Eigen::Matrix3d R1, R2;
  Eigen::Vector3d t;

  // Recover the 4 motion hypotheses
  DecomposeE(E21, R1, R2, t);

  // Reconstruct with the 4 hyphoteses and check
  std::vector<Eigen::Vector3d> vP3D1, vP3D2, vP3D3, vP3D4;
  std::vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3,
      vbTriangulated4;
  float parallax1, parallax2, parallax3, parallax4;

  int nGood1 = CheckRT(R1, t, reference_keypoints_, current_keypoints_,
                       index_matches_, vbMatchesInliers, K, vP3D1,
                       4.0 * sigma2_, vbTriangulated1, parallax1);
  int nGood2 = CheckRT(R2, t, reference_keypoints_, current_keypoints_,
                       index_matches_, vbMatchesInliers, K, vP3D2,
                       4.0 * sigma2_, vbTriangulated2, parallax2);
  int nGood3 = CheckRT(R1, -t, reference_keypoints_, current_keypoints_,
                       index_matches_, vbMatchesInliers, K, vP3D3,
                       4.0 * sigma2_, vbTriangulated3, parallax3);
  int nGood4 = CheckRT(R2, -t, reference_keypoints_, current_keypoints_,
                       index_matches_, vbMatchesInliers, K, vP3D4,
                       4.0 * sigma2_, vbTriangulated4, parallax4);

  int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

  int nMinGood = max(static_cast<int>(0.9 * N), minTriangulated);

  int nsimilar = 0;
  if (nGood1 > 0.7 * maxGood) nsimilar++;
  if (nGood2 > 0.7 * maxGood) nsimilar++;
  if (nGood3 > 0.7 * maxGood) nsimilar++;
  if (nGood4 > 0.7 * maxGood) nsimilar++;

  // If there is not a clear winner or not enough triangulated points reject
  // initialization
  if (maxGood < nMinGood || nsimilar > 1) {
    return false;
  }

  // If best reconstruction has enough parallax initialize
  if (maxGood == nGood1) {
    if (parallax1 > minParallax) {
      vP3D.swap(vP3D1);
      is_triangulated = vbTriangulated1;

      R21 = R1;
      t21 = t;
      return true;
    }
  } else if (maxGood == nGood2) {
    if (parallax2 > minParallax) {
      vP3D.swap(vP3D2);
      is_triangulated = vbTriangulated2;

      R21 = R2;
      t21 = t;
      return true;
    }
  } else if (maxGood == nGood3) {
    if (parallax3 > minParallax) {
      vP3D.swap(vP3D3);
      is_triangulated = vbTriangulated3;

      R21 = R1;
      t21 = -t;
      return true;
    }
  } else if (maxGood == nGood4) {
    if (parallax4 > minParallax) {
      vP3D.swap(vP3D4);
      is_triangulated = vbTriangulated4;

      R21 = R2;
      t21 = -t;
      return true;
    }
  }
  return false;
}

bool Initializer::ReconstructH(vector<bool>& vbMatchesInliers,
                               Eigen::Matrix3d& H21, Eigen::Matrix3d& K,
                               Eigen::Matrix3d& R21, Eigen::Vector3d& t21,
                               std::vector<Eigen::Vector3d>& vP3D,
                               vector<bool>& is_triangulated, float minParallax,
                               int minTriangulated) {
  int N = 0;
  for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
    if (vbMatchesInliers[i]) N++;

  // We recover 8 motion hypotheses using the method of Faugeras et al.
  // Motion and structure from motion in a piecewise planar environment.
  // International Journal of Pattern Recognition and Artificial Intelligence,
  // 1988

  Eigen::Matrix3d A = K.inverse() * H21 * K;

  Eigen::Matrix3d U, V;
  Eigen::Vector3d singular;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  U = svd.matrixU();
  V = svd.matrixV();
  singular = svd.singularValues();

  float s = U.determinant() * V.determinant();

  float d1 = singular(0);
  float d2 = singular(1);
  float d3 = singular(2);

  if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001) {
    return false;
  }

  std::vector<Eigen::Matrix3d> vR;
  std::vector<Eigen::Vector3d> vt, vn;
  vR.reserve(8);
  vt.reserve(8);
  vn.reserve(8);

  // n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
  float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
  float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
  float x1[] = {aux1, aux1, -aux1, -aux1};
  float x3[] = {aux3, -aux3, aux3, -aux3};

  // case d'=d2
  float aux_stheta =
      sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

  float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
  float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

  for (int i = 0; i < 4; i++) {
    Eigen::Matrix3d Rp = Eigen::Matrix3d::Identity();
    Rp(0, 0) = ctheta;
    Rp(0, 2) = -stheta[i];
    Rp(2, 0) = stheta[i];
    Rp(2, 2) = ctheta;

    Eigen::Matrix3d R = s * U * Rp * V.transpose();
    vR.push_back(R);

    Eigen::Vector3d tp;
    tp << x1[i], 0, -x3[i];
    tp *= d1 - d3;

    Eigen::Vector3d t = U * tp;
    vt.push_back(t / t.norm());

    Eigen::Vector3d np;
    np << x1[i], 0, x3[i];

    Eigen::Vector3d n = V * np;
    if (n(2) < 0) n = -n;
    vn.push_back(n);
  }

  // case d'=-d2
  float aux_sphi =
      sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

  float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
  float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

  for (int i = 0; i < 4; i++) {
    Eigen::Matrix3d Rp = Eigen::Matrix3d::Identity();
    Rp(0, 0) = cphi;
    Rp(0, 2) = sphi[i];
    Rp(1, 1) = -1;
    Rp(2, 0) = sphi[i];
    Rp(2, 2) = -cphi;

    Eigen::Matrix3d R = s * U * Rp * V.transpose();
    vR.push_back(R);

    Eigen::Vector3d tp;
    tp << x1[i], 0, x3[i];
    tp *= d1 + d3;

    Eigen::Vector3d t = U * tp;
    vt.push_back(t / t.norm());

    Eigen::Vector3d np;
    np << x1[i], 0, x3[i];
    Eigen::Vector3d n = V * np;
    if (n(2) < 0) n = -n;
    vn.push_back(n);
  }

  int bestGood = 0;
  int secondBestGood = 0;
  int bestSolutionIdx = -1;
  float bestParallax = -1;
  std::vector<Eigen::Vector3d> bestP3D;
  std::vector<bool> bestTriangulated;

  // Instead of applying the visibility constraints proposed in the Faugeras'
  // paper (which could fail for points seen with low parallax) We reconstruct
  // all hypotheses and check in terms of triangulated points and parallax
  for (size_t i = 0; i < 8; i++) {
    float parallaxi;
    std::vector<Eigen::Vector3d> vP3Di;
    std::vector<bool> vbTriangulatedi;
    int nGood = CheckRT(vR[i], vt[i], reference_keypoints_, current_keypoints_,
                        index_matches_, vbMatchesInliers, K, vP3Di,
                        4.0 * sigma2_, vbTriangulatedi, parallaxi);

    if (nGood > bestGood) {
      secondBestGood = bestGood;
      bestGood = nGood;
      bestSolutionIdx = i;
      bestParallax = parallaxi;
      bestP3D.swap(vP3Di);
      bestTriangulated = vbTriangulatedi;
    } else if (nGood > secondBestGood) {
      secondBestGood = nGood;
    }
  }

  if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax &&
      bestGood > minTriangulated && bestGood > 0.9 * N) {
    R21 = vR[bestSolutionIdx];
    t21 = vt[bestSolutionIdx];
    vP3D.swap(bestP3D);
    is_triangulated = bestTriangulated;

    return true;
  }

  return false;
}

// https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_7_2-triangulation.pdf
void Initializer::Triangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                              const Eigen::Matrix<double, 3, 4>& P1,
                              const Eigen::Matrix<double, 3, 4>& P2,
                              Eigen::Vector3d& x3D) {
  Eigen::Matrix4d A;

  A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
  A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
  A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
  A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix4d> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d X = svd.matrixV().col(3);
  x3D = X.block<3, 1>(0, 0) / X(3);
}

void Initializer::Normalize(const vector<cv::KeyPoint>& vKeys,
                            vector<cv::Point2f>& vNormalizedPoints,
                            Eigen::Matrix3d& T) {
  float meanX = 0;
  float meanY = 0;
  const int N = vKeys.size();

  vNormalizedPoints.resize(N);

  for (int i = 0; i < N; i++) {
    meanX += vKeys[i].pt.x;
    meanY += vKeys[i].pt.y;
  }

  meanX = meanX / N;
  meanY = meanY / N;

  float meanDevX = 0;
  float meanDevY = 0;

  for (int i = 0; i < N; i++) {
    vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
    vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

    meanDevX += fabs(vNormalizedPoints[i].x);
    meanDevY += fabs(vNormalizedPoints[i].y);
  }

  meanDevX = meanDevX / N;
  meanDevY = meanDevY / N;

  float sX = 1.0 / meanDevX;
  float sY = 1.0 / meanDevY;

  for (int i = 0; i < N; i++) {
    vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
    vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
  }

  T = Eigen::Matrix3d::Identity();
  T << sX, 0, -meanX * sX, 0, sY, -meanY * sY, 0, 0, 1.;
}

int Initializer::CheckRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                         const vector<cv::KeyPoint>& vKeys1,
                         const vector<cv::KeyPoint>& vKeys2,
                         const vector<Match>& matches,
                         vector<bool>& vbMatchesInliers,
                         const Eigen::Matrix3d& K,
                         std::vector<Eigen::Vector3d>& vP3D, float th2,
                         vector<bool>& vbGood, float& parallax) {
  // Calibration parameters
  const float fx = K(0, 0);
  const float fy = K(1, 1);
  const float cx = K(0, 2);
  const float cy = K(1, 2);

  vbGood = vector<bool>(vKeys1.size(), false);
  vP3D.resize(vKeys1.size());

  vector<float> vCosParallax;
  vCosParallax.reserve(vKeys1.size());

  // Camera 1 Projection Matrix K[I|0]
  Eigen::Matrix<double, 3, 4> P1 = Eigen::Matrix<double, 3, 4>::Zero();
  P1.block<3, 3>(0, 0) = K;

  Eigen::Vector3d O1 = Eigen::Vector3d::Zero();

  // Camera 2 Projection Matrix K[R|t]
  Eigen::Matrix<double, 3, 4> P2 = Eigen::Matrix<double, 3, 4>::Zero();
  P2.block<3, 3>(0, 0) = R;
  P2.block<3, 1>(0, 3) = t;
  P2 = K * P2;

  Eigen::Vector3d O2 = -R.transpose() * t;

  int nGood = 0;

  for (size_t i = 0, iend = matches.size(); i < iend; i++) {
    if (!vbMatchesInliers[i]) continue;

    const cv::KeyPoint& kp1 = vKeys1[matches[i].first];
    const cv::KeyPoint& kp2 = vKeys2[matches[i].second];
    Eigen::Vector3d p3dC1;

    Triangulate(kp1, kp2, P1, P2, p3dC1);

    if (!isfinite(p3dC1(0)) || !isfinite(p3dC1(1)) || !isfinite(p3dC1(2))) {
      vbGood[matches[i].first] = false;
      continue;
    }

    // Check parallax
    Eigen::Vector3d normal1 = p3dC1 - O1;
    float dist1 = normal1.norm();

    Eigen::Vector3d normal2 = p3dC1 - O2;
    float dist2 = normal2.norm();

    float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

    // Check depth in front of first camera (only if enough parallax, as
    // "infinite" points can easily go to negative depth)
    if (p3dC1(2) <= 0 && cosParallax < 0.99998) continue;

    // Check depth in front of second camera (only if enough parallax, as
    // "infinite" points can easily go to negative depth)
    Eigen::Vector3d p3dC2 = R * p3dC1 + t;

    if (p3dC2(2) <= 0 && cosParallax < 0.99998) continue;

    // Check reprojection error in first image
    float im1x, im1y;
    float invZ1 = 1.0 / p3dC1(2);
    im1x = fx * p3dC1(0) * invZ1 + cx;
    im1y = fy * p3dC1(1) * invZ1 + cy;

    float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) +
                         (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

    if (squareError1 > th2) continue;

    // Check reprojection error in second image
    float im2x, im2y;
    float invZ2 = 1.0 / p3dC2(2);
    im2x = fx * p3dC2(0) * invZ2 + cx;
    im2y = fy * p3dC2(1) * invZ2 + cy;

    float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) +
                         (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

    if (squareError2 > th2) continue;

    vCosParallax.push_back(cosParallax);
    vP3D[matches[i].first] = Eigen::Vector3d(p3dC1(0), p3dC1(1), p3dC1(2));
    nGood++;

    if (cosParallax < 0.99998) vbGood[matches[i].first] = true;
  }

  if (nGood > 0) {
    sort(vCosParallax.begin(), vCosParallax.end());

    size_t index = min(50, int(vCosParallax.size() - 1));
    parallax = acos(vCosParallax[index]) * 180 / CV_PI;
  } else
    parallax = 0;

  return nGood;
}

void Initializer::DecomposeE(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1,
                             Eigen::Matrix3d& R2, Eigen::Vector3d& t) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U, V;
  U = svd.matrixU();
  V = svd.matrixV();

  Eigen::Matrix3d W, Z;
  W << 0, -1., 0, 1., 0, 0, 0, 0, 1.;
  Z << 0, 1., 0, -1., 0, 0, 0, 0, 0;

  R1 = U * W * V.transpose();
  R2 = U * W.transpose() * V.transpose();

  t = U.col(2);
  t = t / t.norm();

  if (R1.determinant() < 0) R1 = -R1;

  if (R2.determinant() < 0) R2 = -R2;
}

}  // namespace ORB_SLAM2
