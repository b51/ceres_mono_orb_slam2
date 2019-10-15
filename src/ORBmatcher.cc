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

#include "ORBmatcher.h"

#include <glog/logging.h>
#include <limits.h>
#include <stdint-gcc.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "lib/DBoW2/DBoW2/FeatureVector.h"

using namespace std;

namespace ORB_SLAM2 {

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

int ORBmatcher::SearchByProjection(Frame& F,
                                   const vector<MapPoint*>& vpMapPoints,
                                   const float th) {
  int nmatches = 0;

  const bool bFactor = th != 1.0;

  for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
    MapPoint* pMP = vpMapPoints[iMP];
    if (!pMP->is_track_in_view_) continue;

    if (pMP->isBad()) continue;

    const int& nPredictedLevel = pMP->track_scale_level_;

    // The size of the window will depend on the viewing direction
    float r = RadiusByViewingCos(pMP->track_view_cos_);

    if (bFactor) r *= th;

    const vector<size_t> vIndices =
        F.GetFeaturesInArea(pMP->track_proj_x_, pMP->track_proj_y_,
                            r * F.scale_factors_[nPredictedLevel],
                            nPredictedLevel - 1, nPredictedLevel);

    if (vIndices.empty()) continue;

    const cv::Mat MPdescriptor = pMP->GetDescriptor();

    int bestDist = 256;
    int bestLevel = -1;
    int bestDist2 = 256;
    int bestLevel2 = -1;
    int bestIdx = -1;

    // Get best and second matches with near keypoints
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      if (F.map_points_[idx])
        if (F.map_points_[idx]->Observations() > 0) continue;

      if (F.mvuRight[idx] > 0) {
        const float er = fabs(pMP->track_proj_x_r_ - F.mvuRight[idx]);
        if (er > r * F.scale_factors_[nPredictedLevel]) continue;
      }

      const cv::Mat& d = F.descriptors_.row(idx);

      const int dist = DescriptorDistance(MPdescriptor, d);

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestLevel2 = bestLevel;
        bestLevel = F.undistort_keypoints_[idx].octave;
        bestIdx = idx;
      } else if (dist < bestDist2) {
        bestLevel2 = F.undistort_keypoints_[idx].octave;
        bestDist2 = dist;
      }
    }

    // Apply ratio to second match (only if best and second are in the same
    // scale level)
    if (bestDist <= TH_HIGH) {
      if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2) continue;

      F.map_points_[bestIdx] = pMP;
      nmatches++;
    }
  }

  // LOG(INFO) << "nmatches projection with map points: " << nmatches;
  return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float& viewCos) {
  if (viewCos > 0.998)
    return 2.5;
  else
    return 4.0;
}

bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint& kp1,
                                       const cv::KeyPoint& kp2,
                                       const Eigen::Matrix3d& F12,
                                       const KeyFrame* pKF2) {
  // Epipolar line in second image l = x1'F12 = [a b c]
  const float a = kp1.pt.x * F12(0, 0) +
                  kp1.pt.y * F12(1, 0) + F12(2, 0);
  const float b = kp1.pt.x * F12(0, 1) +
                  kp1.pt.y * F12(1, 1) + F12(2, 1);
  const float c = kp1.pt.x * F12(0, 2) +
                  kp1.pt.y * F12(1, 2) + F12(2, 2);

  const float num = a * kp2.pt.x + b * kp2.pt.y + c;

  const float den = a * a + b * b;

  if (den == 0) return false;

  const float dsqr = num * num / den;

  return dsqr < 3.84 * pKF2->level_sigma2s_[kp2.octave];
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF, Frame& F,
                            vector<MapPoint*>& vpMapPointMatches) {
  const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

  vpMapPointMatches = vector<MapPoint*>(F.N_, static_cast<MapPoint*>(nullptr));

  const DBoW2::FeatureVector& vFeatVecKF = pKF->feature_vector_;

  int nmatches = 0;

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  // We perform the matching over ORB that belong to the same vocabulary node
  // (at a certain level)
  DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
  DBoW2::FeatureVector::const_iterator Fit = F.feature_vector_.begin();
  DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
  DBoW2::FeatureVector::const_iterator Fend = F.feature_vector_.end();

  while (KFit != KFend && Fit != Fend) {
    if (KFit->first == Fit->first) {
      const vector<unsigned int> vIndicesKF = KFit->second;
      const vector<unsigned int> vIndicesF = Fit->second;

      for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
        const unsigned int realIdxKF = vIndicesKF[iKF];

        MapPoint* pMP = vpMapPointsKF[realIdxKF];

        if (!pMP) continue;

        if (pMP->isBad()) continue;

        const cv::Mat& dKF = pKF->descriptors_.row(realIdxKF);

        int bestDist1 = 256;
        int bestIdxF = -1;
        int bestDist2 = 256;

        for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
          const unsigned int realIdxF = vIndicesF[iF];

          if (vpMapPointMatches[realIdxF]) continue;

          const cv::Mat& dF = F.descriptors_.row(realIdxF);

          const int dist = DescriptorDistance(dKF, dF);

          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdxF = realIdxF;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        if (bestDist1 <= TH_LOW) {
          if (static_cast<float>(bestDist1) <
              mfNNratio * static_cast<float>(bestDist2)) {
            vpMapPointMatches[bestIdxF] = pMP;

            const cv::KeyPoint& kp = pKF->undistort_keypoints_[realIdxKF];

            if (mbCheckOrientation) {
              float rot = kp.angle - F.keypoints_[bestIdxF].angle;
              if (rot < 0.0) rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH) bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(bestIdxF);
            }
            nmatches++;
          }
        }
      }

      KFit++;
      Fit++;
    } else if (KFit->first < Fit->first) {
      KFit = vFeatVecKF.lower_bound(Fit->first);
    } else {
      Fit = F.feature_vector_.lower_bound(KFit->first);
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
        nmatches--;
      }
    }
  }

  return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, const Eigen::Matrix4d& Scw,
                                   const vector<MapPoint*>& vpPoints,
                                   vector<MapPoint*>& vpMatched, int th) {
  // Get Calibration Parameters for later projection
  const float& fx = pKF->fx_;
  const float& fy = pKF->fy_;
  const float& cx = pKF->cx_;
  const float& cy = pKF->cy_;

  // Decompose Scw
  Eigen::Matrix3d sRcw = Scw.block<3, 3>(0, 0);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  Eigen::Matrix3d Rcw = sRcw / scw;
  Eigen::Vector3d tcw = Scw.block<3, 1>(0, 3) / scw;
  Eigen::Vector3d Ow = -Rcw.transpose() * tcw;

  // Set of MapPoints already found in the KeyFrame
  set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
  spAlreadyFound.erase(static_cast<MapPoint*>(nullptr));

  int nmatches = 0;

  // For each Candidate MapPoint Project and Match
  for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++) {
    MapPoint* pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP)) continue;

    // Get 3D Coords.
    Eigen::Vector3d p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    Eigen::Vector3d p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc[2] < 0.0) continue;

    // Project into Image
    const float invz = 1 / p3Dc[2];
    const float x = p3Dc[0] * invz;
    const float y = p3Dc[1] * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v)) continue;

    // Depth must be inside the scale invariance region of the point
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    Eigen::Vector3d PO = p3Dw - Ow;
    const float dist = PO.norm();

    if (dist < minDistance || dist > maxDistance) continue;

    // Viewing angle must be less than 60 deg
    Eigen::Vector3d Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist) continue;

    int nPredictedLevel = pMP->PredictScale(dist, pKF);

    // Search in a radius
    const float radius = th * pKF->scale_factors_[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = 256;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;
      if (vpMatched[idx]) continue;

      const int& kpLevel = pKF->undistort_keypoints_[idx].octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel) continue;

      const cv::Mat& dKF = pKF->descriptors_.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    if (bestDist <= TH_LOW) {
      vpMatched[bestIdx] = pMP;
      nmatches++;
    }
  }

  return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame& F1, Frame& F2,
                                        vector<cv::Point2f>& vbPrevMatched,
                                        vector<int>& vnMatches12,
                                        int windowSize) {
  int nmatches = 0;
  vnMatches12 = vector<int>(F1.undistort_keypoints_.size(), -1);

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) {
    rotHist[i].reserve(500);
  }
  const float factor = 1.0f / HISTO_LENGTH;

  vector<int> vMatchedDistance(F2.undistort_keypoints_.size(), INT_MAX);
  vector<int> vnMatches21(F2.undistort_keypoints_.size(), -1);

  for (size_t i1 = 0, iend1 = F1.undistort_keypoints_.size(); i1 < iend1;
       i1++) {
    cv::KeyPoint kp1 = F1.undistort_keypoints_[i1];
    int level1 = kp1.octave;
    if (level1 > 0) {
      continue;
    }

    vector<size_t> vIndices2 = F2.GetFeaturesInArea(
        vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1);

    if (vIndices2.empty()) {
      continue;
    }

    cv::Mat d1 = F1.descriptors_.row(i1);

    int bestDist = INT_MAX;
    int bestDist2 = INT_MAX;
    int bestIdx2 = -1;

    for (vector<size_t>::iterator vit = vIndices2.begin();
         vit != vIndices2.end(); vit++) {
      size_t i2 = *vit;

      cv::Mat d2 = F2.descriptors_.row(i2);

      int dist = DescriptorDistance(d1, d2);

      if (vMatchedDistance[i2] <= dist) continue;

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestIdx2 = i2;
      } else if (dist < bestDist2) {
        bestDist2 = dist;
      }
    }

    if (bestDist <= TH_LOW) {
      if (bestDist < (float)bestDist2 * mfNNratio) {
        if (vnMatches21[bestIdx2] >= 0) {
          vnMatches12[vnMatches21[bestIdx2]] = -1;
          nmatches--;
        }
        vnMatches12[i1] = bestIdx2;
        vnMatches21[bestIdx2] = i1;
        vMatchedDistance[bestIdx2] = bestDist;
        nmatches++;

        if (mbCheckOrientation) {
          float rot = F1.undistort_keypoints_[i1].angle -
                      F2.undistort_keypoints_[bestIdx2].angle;
          if (rot < 0.0) rot += 360.0f;
          int bin = round(rot * factor);
          if (bin == HISTO_LENGTH) bin = 0;
          assert(bin >= 0 && bin < HISTO_LENGTH);
          rotHist[bin].push_back(i1);
        }
      }
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        int idx1 = rotHist[i][j];
        if (vnMatches12[idx1] >= 0) {
          vnMatches12[idx1] = -1;
          nmatches--;
        }
      }
    }
  }

  // Update prev matched
  for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
    if (vnMatches12[i1] >= 0)
      vbPrevMatched[i1] = F2.undistort_keypoints_[vnMatches12[i1]].pt;

  return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF1, KeyFrame* pKF2,
                            vector<MapPoint*>& vpMatches12) {
  const vector<cv::KeyPoint>& vKeysUn1 = pKF1->undistort_keypoints_;
  const DBoW2::FeatureVector& vFeatVec1 = pKF1->feature_vector_;
  const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
  const cv::Mat& Descriptors1 = pKF1->descriptors_;

  const vector<cv::KeyPoint>& vKeysUn2 = pKF2->undistort_keypoints_;
  const DBoW2::FeatureVector& vFeatVec2 = pKF2->feature_vector_;
  const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
  const cv::Mat& Descriptors2 = pKF2->descriptors_;

  vpMatches12 =
      vector<MapPoint*>(vpMapPoints1.size(), static_cast<MapPoint*>(nullptr));
  vector<bool> vbMatched2(vpMapPoints2.size(), false);

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);

  const float factor = 1.0f / HISTO_LENGTH;

  int nmatches = 0;

  DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
  DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
  DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
  DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

  while (f1it != f1end && f2it != f2end) {
    if (f1it->first == f2it->first) {
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];

        MapPoint* pMP1 = vpMapPoints1[idx1];
        if (!pMP1) continue;
        if (pMP1->isBad()) continue;

        const cv::Mat& d1 = Descriptors1.row(idx1);

        int bestDist1 = 256;
        int bestIdx2 = -1;
        int bestDist2 = 256;

        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          const size_t idx2 = f2it->second[i2];

          MapPoint* pMP2 = vpMapPoints2[idx2];

          if (vbMatched2[idx2] || !pMP2) continue;

          if (pMP2->isBad()) continue;

          const cv::Mat& d2 = Descriptors2.row(idx2);

          int dist = DescriptorDistance(d1, d2);

          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdx2 = idx2;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        if (bestDist1 < TH_LOW) {
          if (static_cast<float>(bestDist1) <
              mfNNratio * static_cast<float>(bestDist2)) {
            vpMatches12[idx1] = vpMapPoints2[bestIdx2];
            vbMatched2[bestIdx2] = true;

            if (mbCheckOrientation) {
              float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
              if (rot < 0.0) rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH) bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(idx1);
            }
            nmatches++;
          }
        }
      }

      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vpMatches12[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
        nmatches--;
      }
    }
  }

  return nmatches;
}

int ORBmatcher::SearchForTriangulation(
    KeyFrame* pKF1, KeyFrame* pKF2, const Eigen::Matrix3d& F12,
    vector<pair<size_t, size_t> >& vMatchedPairs, const bool bOnlyStereo) {
  const DBoW2::FeatureVector& vFeatVec1 = pKF1->feature_vector_;
  const DBoW2::FeatureVector& vFeatVec2 = pKF2->feature_vector_;

  // Compute epipole in second image
  Eigen::Vector3d Cw = pKF1->GetCameraCenter();
  Eigen::Matrix3d R2w = pKF2->GetRotation();
  Eigen::Vector3d t2w = pKF2->GetTranslation();
  Eigen::Vector3d C2 = R2w * Cw + t2w;
  const float invz = 1.0f / C2[2];
  const float ex = pKF2->fx_ * C2[0] * invz + pKF2->cx_;
  const float ey = pKF2->fy_ * C2[1] * invz + pKF2->cy_;

  // Find matches between not tracked keypoints
  // Matching speed-up by ORB Vocabulary
  // Compare only ORB that share the same node

  int nmatches = 0;
  vector<bool> vbMatched2(pKF2->N_, false);
  vector<int> vMatches12(pKF1->N_, -1);

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);

  const float factor = 1.0f / HISTO_LENGTH;

  DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
  DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
  DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
  DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

  while (f1it != f1end && f2it != f2end) {
    if (f1it->first == f2it->first) {
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];

        MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

        // If there is already a MapPoint skip
        if (pMP1) continue;

        const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;

        if (bOnlyStereo)
          if (!bStereo1) continue;

        const cv::KeyPoint& kp1 = pKF1->undistort_keypoints_[idx1];

        const cv::Mat& d1 = pKF1->descriptors_.row(idx1);

        int bestDist = TH_LOW;
        int bestIdx2 = -1;

        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          size_t idx2 = f2it->second[i2];

          MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

          // If we have already matched or there is a MapPoint skip
          if (vbMatched2[idx2] || pMP2) continue;

          const bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

          if (bOnlyStereo)
            if (!bStereo2) continue;

          const cv::Mat& d2 = pKF2->descriptors_.row(idx2);

          const int dist = DescriptorDistance(d1, d2);

          if (dist > TH_LOW || dist > bestDist) continue;

          const cv::KeyPoint& kp2 = pKF2->undistort_keypoints_[idx2];

          if (!bStereo1 && !bStereo2) {
            const float distex = ex - kp2.pt.x;
            const float distey = ey - kp2.pt.y;
            if (distex * distex + distey * distey <
                100 * pKF2->scale_factors_[kp2.octave])
              continue;
          }

          if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2)) {
            bestIdx2 = idx2;
            bestDist = dist;
          }
        }

        if (bestIdx2 >= 0) {
          const cv::KeyPoint& kp2 = pKF2->undistort_keypoints_[bestIdx2];
          vMatches12[idx1] = bestIdx2;
          nmatches++;

          if (mbCheckOrientation) {
            float rot = kp1.angle - kp2.angle;
            if (rot < 0.0) rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH) bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(idx1);
          }
        }
      }

      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = vFeatVec1.lower_bound(f2it->first);
    } else {
      f2it = vFeatVec2.lower_bound(f1it->first);
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3) continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        vMatches12[rotHist[i][j]] = -1;
        nmatches--;
      }
    }
  }

  vMatchedPairs.clear();
  vMatchedPairs.reserve(nmatches);

  for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
    if (vMatches12[i] < 0) continue;
    vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
  }

  return nmatches;
}

int ORBmatcher::Fuse(KeyFrame* pKF, const vector<MapPoint*>& vpMapPoints,
                     const float th) {
  Eigen::Matrix3d Rcw = pKF->GetRotation();
  Eigen::Vector3d tcw = pKF->GetTranslation();

  const float& fx = pKF->fx_;
  const float& fy = pKF->fy_;
  const float& cx = pKF->cx_;
  const float& cy = pKF->cy_;
  // const float& bf = pKF->bf_;

  Eigen::Vector3d Ow = pKF->GetCameraCenter();

  int nFused = 0;

  const int nMPs = vpMapPoints.size();

  for (int i = 0; i < nMPs; i++) {
    MapPoint* pMP = vpMapPoints[i];

    if (!pMP) continue;

    if (pMP->isBad() || pMP->IsInKeyFrame(pKF)) continue;

    Eigen::Vector3d p3Dw = pMP->GetWorldPos();
    Eigen::Vector3d p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc[2] < 0.0f) continue;

    const float invz = 1 / p3Dc[2];
    const float x = p3Dc[0] * invz;
    const float y = p3Dc[1] * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v)) continue;

    // const float ur = u - bf * invz;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    Eigen::Vector3d PO = p3Dw - Ow;
    const float dist3D = PO.norm();

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Viewing angle must be less than 60 deg
    Eigen::Vector3d Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D) continue;

    int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->scale_factors_[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = 256;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint& kp = pKF->undistort_keypoints_[idx];

      const int& kpLevel = kp.octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel) continue;

      const float& kpx = kp.pt.x;
      const float& kpy = kp.pt.y;
      const float ex = u - kpx;
      const float ey = v - kpy;
      const float e2 = ex * ex + ey * ey;

      if (e2 * pKF->inv_level_sigma2s_[kpLevel] > 5.99) continue;

      const cv::Mat& dKF = pKF->descriptors_.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW) {
      MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad()) {
          if (pMPinKF->Observations() > pMP->Observations())
            pMP->Replace(pMPinKF);
          else
            pMPinKF->Replace(pMP);
        }
      } else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int ORBmatcher::Fuse(KeyFrame* pKF, Eigen::Matrix4d Scw,
                     const vector<MapPoint*>& vpPoints, float th,
                     vector<MapPoint*>& vpReplacePoint) {
  // Get Calibration Parameters for later projection
  const float& fx = pKF->fx_;
  const float& fy = pKF->fy_;
  const float& cx = pKF->cx_;
  const float& cy = pKF->cy_;

  // Decompose Scw
  Eigen::Matrix3d sRcw = Scw.block<3, 3>(0, 0);
  const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
  Eigen::Matrix3d Rcw = sRcw / scw;
  Eigen::Vector3d tcw = Scw.block<3, 1>(0, 3) / scw;
  Eigen::Vector3d Ow = -Rcw.transpose() * tcw;

  // Set of MapPoints already found in the KeyFrame
  const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

  int nFused = 0;

  const int nPoints = vpPoints.size();

  // For each candidate MapPoint project and match
  for (int iMP = 0; iMP < nPoints; iMP++) {
    MapPoint* pMP = vpPoints[iMP];

    // Discard Bad MapPoints and already found
    if (pMP->isBad() || spAlreadyFound.count(pMP)) continue;

    // Get 3D Coords.
    Eigen::Vector3d p3Dw = pMP->GetWorldPos();

    // Transform into Camera Coords.
    Eigen::Vector3d p3Dc = Rcw * p3Dw + tcw;

    // Depth must be positive
    if (p3Dc[2] < 0.0f) continue;

    // Project into Image
    const float invz = 1.0 / p3Dc[2];
    const float x = p3Dc[0] * invz;
    const float y = p3Dc[1] * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF->IsInImage(u, v)) continue;

    // Depth must be inside the scale pyramid of the image
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    Eigen::Vector3d PO = p3Dw - Ow;
    const float dist3D = PO.norm();

    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Viewing angle must be less than 60 deg
    Eigen::Vector3d Pn = pMP->GetNormal();

    if (PO.dot(Pn) < 0.5 * dist3D) continue;

    // Compute predicted scale level
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

    // Search in a radius
    const float radius = th * pKF->scale_factors_[nPredictedLevel];

    const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius

    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = INT_MAX;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin();
         vit != vIndices.end(); vit++) {
      const size_t idx = *vit;
      const int& kpLevel = pKF->undistort_keypoints_[idx].octave;

      if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel) continue;

      const cv::Mat& dKF = pKF->descriptors_.row(idx);

      int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (bestDist <= TH_LOW) {
      MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
      if (pMPinKF) {
        if (!pMPinKF->isBad()) vpReplacePoint[iMP] = pMPinKF;
      } else {
        pMP->AddObservation(pKF, bestIdx);
        pKF->AddMapPoint(pMP, bestIdx);
      }
      nFused++;
    }
  }

  return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2,
                             vector<MapPoint*>& vpMatches12, const float& s12,
                             const Eigen::Matrix3d& R12,
                             const Eigen::Vector3d& t12, const float th) {
  const float& fx = pKF1->fx_;
  const float& fy = pKF1->fy_;
  const float& cx = pKF1->cx_;
  const float& cy = pKF1->cy_;

  // Camera 1 from world
  Eigen::Matrix3d R1w = pKF1->GetRotation();
  Eigen::Vector3d t1w = pKF1->GetTranslation();

  // Camera 2 from world
  Eigen::Matrix3d R2w = pKF2->GetRotation();
  Eigen::Vector3d t2w = pKF2->GetTranslation();

  // Transformation between cameras
  Eigen::Matrix3d sR12 = s12 * R12;
  Eigen::Matrix3d sR21 = (1.0 / s12) * R12.transpose();
  Eigen::Vector3d t21 = -sR21 * t12;

  const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
  const int N1 = vpMapPoints1.size();

  const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
  const int N2 = vpMapPoints2.size();

  vector<bool> vbAlreadyMatched1(N1, false);
  vector<bool> vbAlreadyMatched2(N2, false);

  for (int i = 0; i < N1; i++) {
    MapPoint* pMP = vpMatches12[i];
    if (pMP) {
      vbAlreadyMatched1[i] = true;
      int idx2 = pMP->GetIndexInKeyFrame(pKF2);
      if (idx2 >= 0 && idx2 < N2) vbAlreadyMatched2[idx2] = true;
    }
  }

  vector<int> vnMatch1(N1, -1);
  vector<int> vnMatch2(N2, -1);

  // Transform from KF1 to KF2 and search
  for (int i1 = 0; i1 < N1; i1++) {
    MapPoint* pMP = vpMapPoints1[i1];

    if (!pMP || vbAlreadyMatched1[i1]) continue;

    if (pMP->isBad()) continue;

    Eigen::Vector3d p3Dw = pMP->GetWorldPos();
    Eigen::Vector3d p3Dc1 = R1w * p3Dw + t1w;
    Eigen::Vector3d p3Dc2 = sR21 * p3Dc1 + t21;

    // Depth must be positive
    if (p3Dc2[2] < 0.0) continue;

    const float invz = 1.0 / p3Dc2[2];
    const float x = p3Dc2[0] * invz;
    const float y = p3Dc2[1] * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF2->IsInImage(u, v)) continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D = p3Dc2.norm();

    // Depth must be inside the scale invariance region
    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Compute predicted octave
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

    // Search in a radius
    const float radius = th * pKF2->scale_factors_[nPredictedLevel];

    const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = INT_MAX;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint& kp = pKF2->undistort_keypoints_[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat& dKF = pKF2->descriptors_.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    if (bestDist <= TH_HIGH) {
      vnMatch1[i1] = bestIdx;
    }
  }

  // Transform from KF2 to KF2 and search
  for (int i2 = 0; i2 < N2; i2++) {
    MapPoint* pMP = vpMapPoints2[i2];

    if (!pMP || vbAlreadyMatched2[i2]) continue;

    if (pMP->isBad()) continue;

    Eigen::Vector3d p3Dw = pMP->GetWorldPos();
    Eigen::Vector3d p3Dc2 = R2w * p3Dw + t2w;
    Eigen::Vector3d p3Dc1 = sR12 * p3Dc2 + t12;

    // Depth must be positive
    if (p3Dc1[2] < 0.0) continue;

    const float invz = 1.0 / p3Dc1[2];
    const float x = p3Dc1[0] * invz;
    const float y = p3Dc1[1] * invz;

    const float u = fx * x + cx;
    const float v = fy * y + cy;

    // Point must be inside the image
    if (!pKF1->IsInImage(u, v)) continue;

    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const float dist3D = p3Dc1.norm();

    // Depth must be inside the scale pyramid of the image
    if (dist3D < minDistance || dist3D > maxDistance) continue;

    // Compute predicted octave
    const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

    // Search in a radius of 2.5*sigma(ScaleLevel)
    const float radius = th * pKF1->scale_factors_[nPredictedLevel];

    const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

    if (vIndices.empty()) continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = pMP->GetDescriptor();

    int bestDist = INT_MAX;
    int bestIdx = -1;
    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      const cv::KeyPoint& kp = pKF1->undistort_keypoints_[idx];

      if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
        continue;

      const cv::Mat& dKF = pKF1->descriptors_.row(idx);

      const int dist = DescriptorDistance(dMP, dKF);

      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    }

    if (bestDist <= TH_HIGH) {
      vnMatch2[i2] = bestIdx;
    }
  }

  // Check agreement
  int nFound = 0;

  for (int i1 = 0; i1 < N1; i1++) {
    int idx2 = vnMatch1[i1];

    if (idx2 >= 0) {
      int idx1 = vnMatch2[idx2];
      if (idx1 == i1) {
        vpMatches12[i1] = vpMapPoints2[idx2];
        nFound++;
      }
    }
  }

  return nFound;
}

int ORBmatcher::SearchByProjection(Frame& CurrentFrame, const Frame& LastFrame,
                                   const float th) {
  int nmatches = 0;

  // Rotation Histogram (to check rotation consistency)
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  const Eigen::Matrix3d Rcw = CurrentFrame.Tcw_.block<3, 3>(0, 0);
  const Eigen::Vector3d tcw = CurrentFrame.Tcw_.block<3, 1>(0, 3);

  // const Eigen::Vector3d twc = -Rcw.transpose() * tcw;
  // const Eigen::Matrix3d Rlw = LastFrame.Tcw_.block<3, 3>(0, 0);
  // const Eigen::Vector3d tlw = LastFrame.Tcw_.block<3, 1>(0, 3);
  // const Eigen::Vector3d tlc = Rlw * twc + tlw;

  for (int i = 0; i < LastFrame.N_; i++) {
    MapPoint* map_point = LastFrame.map_points_[i];

    if (map_point) {
      if (!LastFrame.is_outliers_[i]) {
        // Project
        Eigen::Vector3d x3Dw = map_point->GetWorldPos();
        Eigen::Vector3d x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc[0];
        const float yc = x3Dc[1];
        const float invzc = 1.0 / x3Dc[2];

        if (invzc < 0) continue;

        float u = CurrentFrame.fx_ * xc * invzc + CurrentFrame.cx_;
        float v = CurrentFrame.fy_ * yc * invzc + CurrentFrame.cy_;

        if (u < CurrentFrame.min_x_ || u > CurrentFrame.max_x_) continue;
        if (v < CurrentFrame.min_y_ || v > CurrentFrame.max_y_) continue;

        int nLastOctave = LastFrame.keypoints_[i].octave;

        // Search in a window. Size depends on scale
        float radius = th * CurrentFrame.scale_factors_[nLastOctave];

        vector<size_t> vIndices2;

        vIndices2 = CurrentFrame.GetFeaturesInArea(
            u, v, radius, nLastOctave - 1, nLastOctave + 1);

        if (vIndices2.empty()) continue;

        const cv::Mat dMP = map_point->GetDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                            vend = vIndices2.end();
             vit != vend; vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.map_points_[i2])
            if (CurrentFrame.map_points_[i2]->Observations() > 0) continue;

          const cv::Mat& d = CurrentFrame.descriptors_.row(i2);

          const int dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= TH_HIGH) {
          CurrentFrame.map_points_[bestIdx2] = map_point;
          nmatches++;

          if (mbCheckOrientation) {
            float rot = LastFrame.undistort_keypoints_[i].angle -
                        CurrentFrame.undistort_keypoints_[bestIdx2].angle;
            if (rot < 0.0) rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH) bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
          }
        }
      }
    }
  } // end of for LastFrame.N_

  // Apply rotation consistency
  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.map_points_[rotHist[i][j]] =
              static_cast<MapPoint*>(nullptr);
          nmatches--;
        }
      }
    }
  }

  return nmatches;
}

int ORBmatcher::SearchByProjection(Frame& CurrentFrame, KeyFrame* pKF,
                                   const set<MapPoint*>& sAlreadyFound,
                                   const float th, const int ORBdist) {
  int nmatches = 0;

  const Eigen::Matrix3d Rcw = CurrentFrame.Tcw_.block<3, 3>(0, 0);
  const Eigen::Vector3d tcw = CurrentFrame.Tcw_.block<3, 1>(0, 3);
  const Eigen::Vector3d Ow = -Rcw.transpose() * tcw;

  // Rotation Histogram (to check rotation consistency)
  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++) rotHist[i].reserve(500);
  const float factor = 1.0f / HISTO_LENGTH;

  const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

  for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
    MapPoint* pMP = vpMPs[i];

    if (pMP) {
      if (!pMP->isBad() && !sAlreadyFound.count(pMP)) {
        // Project
        Eigen::Vector3d x3Dw = pMP->GetWorldPos();
        Eigen::Vector3d x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc[0];
        const float yc = x3Dc[1];
        const float invzc = 1.0 / x3Dc[2];

        const float u = CurrentFrame.fx_ * xc * invzc + CurrentFrame.cx_;
        const float v = CurrentFrame.fy_ * yc * invzc + CurrentFrame.cy_;

        if (u < CurrentFrame.min_x_ || u > CurrentFrame.max_x_) continue;
        if (v < CurrentFrame.min_y_ || v > CurrentFrame.max_y_) continue;

        // Compute predicted scale level
        Eigen::Vector3d PO = x3Dw - Ow;
        float dist3D = PO.norm();

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance) continue;

        int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

        // Search in a window
        const float radius = th * CurrentFrame.scale_factors_[nPredictedLevel];

        const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(
            u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

        if (vIndices2.empty()) continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin();
             vit != vIndices2.end(); vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.map_points_[i2]) continue;

          const cv::Mat& d = CurrentFrame.descriptors_.row(i2);

          const int dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= ORBdist) {
          CurrentFrame.map_points_[bestIdx2] = pMP;
          nmatches++;

          if (mbCheckOrientation) {
            float rot = pKF->undistort_keypoints_[i].angle -
                        CurrentFrame.undistort_keypoints_[bestIdx2].angle;
            if (rot < 0.0) rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH) bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
          }
        }
      }
    }
  }

  if (mbCheckOrientation) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.map_points_[rotHist[i][j]] = nullptr;
          nmatches--;
        }
      }
    }
  }

  return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int& ind1,
                                    int& ind2, int& ind3) {
  int max1 = 0;
  int max2 = 0;
  int max3 = 0;

  for (int i = 0; i < L; i++) {
    const int s = histo[i].size();
    if (s > max1) {
      max3 = max2;
      max2 = max1;
      max1 = s;
      ind3 = ind2;
      ind2 = ind1;
      ind1 = i;
    } else if (s > max2) {
      max3 = max2;
      max2 = s;
      ind3 = ind2;
      ind2 = i;
    } else if (s > max3) {
      max3 = s;
      ind3 = i;
    }
  }

  if (max2 < 0.1f * (float)max1) {
    ind2 = -1;
    ind3 = -1;
  } else if (max3 < 0.1f * (float)max1) {
    ind3 = -1;
  }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat& a, const cv::Mat& b) {
  const int* pa = a.ptr<int32_t>();
  const int* pb = b.ptr<int32_t>();

  int dist = 0;

  // a: 1 row, 32 cols = uint8_t * 32
  for (int i = 0; i < 8; i++, pa++, pb++) {
    unsigned int v = *pa ^ *pb;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}

}  // namespace ORB_SLAM2
