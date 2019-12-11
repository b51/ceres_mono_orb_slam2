/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: LoopClosing.cc
 *
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

#include "LoopClosing.h"
#include "CeresOptimizer.h"
#include "MatEigenConverter.h"
#include "ORBmatcher.h"
#include "Sim3Solver.h"

#include <unistd.h>
#include <mutex>
#include <thread>

namespace ORB_SLAM2 {

LoopClosing::LoopClosing(Map* map, KeyFrameDatabase* keyframe_database,
                         ORBVocabulary* orb_vocabulary, const bool is_fix_scale)
    : is_reset_requested_(false),
      is_finish_requested_(false),
      is_finished_(true),
      map_(map),
      keyframe_database_(keyframe_database),
      orb_vocabulary_(orb_vocabulary),
      matched_keyframe_(nullptr),
      last_loop_keyframe_id_(0),
      is_running_global_BA_(false),
      is_finished_global_BA_(true),
      is_stop_global_BA_(false),
      thread_global_BA_(nullptr),
      is_fix_scale_(is_fix_scale),
      full_BA_index_(0) {
  covisibility_consistency_threshold_ = 3;
}

void LoopClosing::SetTracker(Tracking* tracker) { tracker_ = tracker; }

void LoopClosing::SetLocalMapper(LocalMapping* local_mapper) {
  local_mapper_ = local_mapper;
}

void LoopClosing::Run() {
  is_finished_ = false;

  while (true) {
    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // Detect loop candidates and check covisibility consistency
      if (DetectLoop()) {
        // Compute similarity transformation [sR|t]
        // In the stereo/RGBD case s=1
        if (ComputeSim3()) {
          // Perform loop fusion and pose graph optimization
          CorrectLoop();
        }
      }
    }

    ResetIfRequested();

    if (CheckFinish()) break;

    usleep(5000);
  }

  SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_loop_queue_);
  if (keyframe->id_ != 0) {
    loop_keyframe_queue_.push_back(keyframe);
  }
}

bool LoopClosing::CheckNewKeyFrames() {
  unique_lock<mutex> lock(mutex_loop_queue_);
  return (!loop_keyframe_queue_.empty());
}

bool LoopClosing::DetectLoop() {
  {
    unique_lock<mutex> lock(mutex_loop_queue_);
    current_keyframe_ = loop_keyframe_queue_.front();
    loop_keyframe_queue_.pop_front();
    // Avoid that a keyframe can be erased while it is being process by this
    // thread
    current_keyframe_->SetNotErase();
  }

  // If the map contains less than 10 KF or less than 10 KF have passed from
  // last loop detection
  if (current_keyframe_->id_ < last_loop_keyframe_id_ + 10) {
    keyframe_database_->add(current_keyframe_);
    current_keyframe_->SetErase();
    return false;
  }

  // Compute reference BoW similarity score
  // This is the lowest score to a connected keyframe in the covisibility graph
  // We will impose loop candidates to have a higher similarity than this
  const vector<KeyFrame*> connected_keyframes =
      current_keyframe_->GetVectorCovisibleKeyFrames();
  const DBoW2::BowVector& current_bow_vector = current_keyframe_->bow_vector_;
  float minScore = 1;
  for (size_t i = 0; i < connected_keyframes.size(); i++) {
    KeyFrame* keyframe = connected_keyframes[i];
    if (keyframe->isBad()) continue;
    const DBoW2::BowVector& bow_vector = keyframe->bow_vector_;

    float score = orb_vocabulary_->score(current_bow_vector, bow_vector);

    if (score < minScore) minScore = score;
  }

  // Query the database imposing the minimum score
  std::vector<KeyFrame*> candidate_keyframes =
      keyframe_database_->DetectLoopCandidates(current_keyframe_, minScore);

  // If there are no loop candidates, just add new keyframe and return false
  // LOG(INFO) << " candidates keyframes size: " << candidate_keyframes.size();
  if (candidate_keyframes.empty()) {
    keyframe_database_->add(current_keyframe_);
    consistent_groups_.clear();
    current_keyframe_->SetErase();
    return false;
  }

  // For each loop candidate check consistency with previous loop candidates
  // Each candidate expands a covisibility group (keyframes connected to the
  // loop candidate in the covisibility graph) A group is consistent with a
  // previous group if they share at least a keyframe We must detect a
  // consistent loop in several consecutive keyframes to accept it
  enough_consistent_candidates_.clear();

  vector<ConsistentGroup> current_consistent_groups;
  vector<bool> is_consistent_group(consistent_groups_.size(), false);
  for (size_t i = 0, iend = candidate_keyframes.size(); i < iend; i++) {
    KeyFrame* candidate_keyframe = candidate_keyframes[i];

    set<KeyFrame*> candidate_group =
        candidate_keyframe->GetConnectedKeyFrames();
    candidate_group.insert(candidate_keyframe);

    bool is_enough_consistent = false;
    bool is_consistent_for_some_group = false;
    for (size_t iG = 0, iendG = consistent_groups_.size(); iG < iendG; iG++) {
      set<KeyFrame*> previous_group = consistent_groups_[iG].first;

      bool is_consistent = false;
      for (set<KeyFrame*>::iterator sit = candidate_group.begin(),
                                    send = candidate_group.end();
           sit != send; sit++) {
        if (previous_group.count(*sit)) {
          is_consistent = true;
          is_consistent_for_some_group = true;
          break;
        }
      }

      if (is_consistent) {
        int previous_consistency = consistent_groups_[iG].second;
        int n_current_consistency = previous_consistency + 1;
        if (!is_consistent_group[iG]) {
          ConsistentGroup cg =
              make_pair(candidate_group, n_current_consistency);
          current_consistent_groups.push_back(cg);
          // this avoid to include the same group more than once
          is_consistent_group[iG] = true;
        }
        if (n_current_consistency >= covisibility_consistency_threshold_ &&
            !is_enough_consistent) {
          enough_consistent_candidates_.push_back(candidate_keyframe);
          // this avoid to insert the same candidate more than once
          is_enough_consistent = true;
        }
      }
    }

    // If the group is not consistent with any previous group insert with
    // consistency counter set to zero
    if (!is_consistent_for_some_group) {
      ConsistentGroup cg = make_pair(candidate_group, 0);
      current_consistent_groups.push_back(cg);
    }
  }

  // Update Covisibility Consistent Groups
  consistent_groups_ = current_consistent_groups;

  // Add Current Keyframe to database
  keyframe_database_->add(current_keyframe_);

  if (enough_consistent_candidates_.empty()) {
    current_keyframe_->SetErase();
    return false;
  } else {
    return true;
  }

  current_keyframe_->SetErase();
  return false;
}

bool LoopClosing::ComputeSim3() {
  // For each consistent loop candidate we try to compute a Sim3

  const int n_initial_candidates = enough_consistent_candidates_.size();

  // We compute first ORB matches for each candidate
  // If enough matches are found, we setup a Sim3Solver
  ORBmatcher matcher(0.75, true);

  vector<Sim3Solver*> sim3_solvers;
  sim3_solvers.resize(n_initial_candidates);

  vector<vector<MapPoint*> > map_point_matches_vector;
  map_point_matches_vector.resize(n_initial_candidates);

  vector<bool> do_discarded;
  do_discarded.resize(n_initial_candidates);

  int n_candidates = 0;  // candidates with enough matches

  for (int i = 0; i < n_initial_candidates; i++) {
    KeyFrame* keyframe = enough_consistent_candidates_[i];

    // avoid that local mapping erase it while it is being processed in this
    // thread
    keyframe->SetNotErase();

    if (keyframe->isBad()) {
      do_discarded[i] = true;
      continue;
    }

    int nmatches = matcher.SearchByBoW(current_keyframe_, keyframe,
                                       map_point_matches_vector[i]);

    if (nmatches < 20) {
      do_discarded[i] = true;
      continue;
    } else {
      Sim3Solver* pSolver =
          new Sim3Solver(current_keyframe_, keyframe,
                         map_point_matches_vector[i], is_fix_scale_);
      pSolver->SetRansacParameters(0.99, 20, 300);
      sim3_solvers[i] = pSolver;
    }

    n_candidates++;
  }

  bool is_match = false;

  // Perform alternatively RANSAC iterations for each candidate
  // until one is succesful or all fail
  while (n_candidates > 0 && !is_match) {
    for (int i = 0; i < n_initial_candidates; i++) {
      if (do_discarded[i]) {
        continue;
      }

      KeyFrame* keyframe = enough_consistent_candidates_[i];

      // Perform 5 Ransac Iterations
      vector<bool> is_inliers;
      int n_inliers;
      bool is_no_more;

      Sim3Solver* pSolver = sim3_solvers[i];
      Eigen::Matrix4d Scm = pSolver->iterate(5, is_no_more, is_inliers, n_inliers);

      // If Ransac reachs max. iterations discard keyframe
      if (is_no_more) {
        do_discarded[i] = true;
        n_candidates--;
      }

      // If RANSAC returns a Sim3, perform a guided matching and optimize with
      // all correspondences
      if (!Scm.isIdentity()) {
        std::vector<MapPoint*> map_point_matches(
            map_point_matches_vector[i].size(), static_cast<MapPoint*>(nullptr));
        for (size_t j = 0, jend = is_inliers.size(); j < jend; j++) {
          if (is_inliers[j])
            map_point_matches[j] = map_point_matches_vector[i][j];
        }

        Eigen::Matrix3d R = pSolver->GetEstimatedRotation();
        Eigen::Vector3d t = pSolver->GetEstimatedTranslation();
        double s = pSolver->GetEstimatedScale();

        matcher.SearchBySim3(current_keyframe_, keyframe, map_point_matches, s,
                             R, t, 7.5);

        Sophus::Sim3d gScm(Sophus::RxSO3d(s, R), t);

        const int n_inliers = CeresOptimizer::OptimizeSim3(
            current_keyframe_, keyframe, map_point_matches, gScm, 10,
            is_fix_scale_);

        // If optimization is succesful stop ransacs and continue
        if (n_inliers >= 20) {
          is_match = true;
          matched_keyframe_ = keyframe;
          Sophus::Sim3d sophus_gSmw(
              Sophus::RxSO3d(1.0, keyframe->GetRotation()),
              keyframe->GetTranslation());
          sophus_sim3_Scw_ = gScm * sophus_gSmw;
          Scw_ = sophus_sim3_Scw_.matrix();

          current_matched_map_points_ = map_point_matches;
          break;
        }
      }
    }
  }

  if (!is_match) {
    for (int i = 0; i < n_initial_candidates; i++)
      enough_consistent_candidates_[i]->SetErase();
    current_keyframe_->SetErase();
    return false;
  }

  // Retrieve MapPoints seen in Loop Keyframe and neighbors
  std::vector<KeyFrame*> loop_connected_keyframes =
      matched_keyframe_->GetVectorCovisibleKeyFrames();
  loop_connected_keyframes.push_back(matched_keyframe_);
  loop_map_points_.clear();
  for (vector<KeyFrame*>::iterator vit = loop_connected_keyframes.begin();
       vit != loop_connected_keyframes.end(); vit++) {
    KeyFrame* keyframe = *vit;
    vector<MapPoint*> map_point_matches = keyframe->GetMapPointMatches();
    for (size_t i = 0, iend = map_point_matches.size(); i < iend; i++) {
      MapPoint* map_point = map_point_matches[i];
      if (map_point) {
        if (!map_point->isBad() &&
            map_point->loop_point_for_keyframe_ != current_keyframe_->id_) {
          loop_map_points_.push_back(map_point);
          map_point->loop_point_for_keyframe_ = current_keyframe_->id_;
        }
      }
    }
  }

  // Find more matches projecting with the computed Sim3
  matcher.SearchByProjection(current_keyframe_, Scw_, loop_map_points_,
                             current_matched_map_points_, 10);

  // If enough matches accept Loop
  int n_total_matches = 0;
  for (size_t i = 0; i < current_matched_map_points_.size(); i++) {
    if (current_matched_map_points_[i]) {
      n_total_matches++;
    }
  }

  if (n_total_matches >= 40) {
    for (int i = 0; i < n_initial_candidates; i++) {
      if (enough_consistent_candidates_[i] != matched_keyframe_) {
        enough_consistent_candidates_[i]->SetErase();
      }
    }
    return true;
  } else {
    for (int i = 0; i < n_initial_candidates; i++) {
      enough_consistent_candidates_[i]->SetErase();
    }
    current_keyframe_->SetErase();
    return false;
  }
}

void LoopClosing::CorrectLoop() {
  LOG(WARNING) << "Loop detected!" << std::endl;

  // Send a stop signal to Local Mapping
  // Avoid new keyframes are inserted while correcting the loop
  local_mapper_->RequestStop();

  // If a Global Bundle Adjustment is running, abort it
  if (isRunningGBA()) {
    unique_lock<mutex> lock(mutex_global_BA_);
    is_stop_global_BA_ = true;

    full_BA_index_++;

    if (thread_global_BA_) {
      thread_global_BA_->detach();
      delete thread_global_BA_;
    }
  }

  // Wait until Local Mapping has effectively stopped
  while (!local_mapper_->isStopped()) {
    usleep(1000);
  }

  // Ensure current keyframe is updated
  current_keyframe_->UpdateConnections();

  // Retrive keyframes connected to the current keyframe and compute corrected
  // Sim3 pose by propagation
  current_connected_keyframes_ =
      current_keyframe_->GetVectorCovisibleKeyFrames();
  current_connected_keyframes_.push_back(current_keyframe_);

  // KeyFrameAndSim3 corrected_sim3, non_corrected_sim3;

  KeyFrameAndSim3 sophus_corrected_sim3, sophus_non_corrected_sim3;
  sophus_corrected_sim3[current_keyframe_] = sophus_sim3_Scw_;

  Eigen::Matrix4d Twc = current_keyframe_->GetPoseInverse();

  {
    // Get Map Mutex
    unique_lock<mutex> lock(map_->mutex_map_update_);

    for (auto vit = current_connected_keyframes_.begin(),
              vend = current_connected_keyframes_.end();
         vit != vend; vit++) {
      KeyFrame* pKFi = *vit;

      Eigen::Matrix4d Tiw = pKFi->GetPose();

      if (pKFi != current_keyframe_) {
        Eigen::Matrix4d Tic = Tiw * Twc;
        Eigen::Matrix3d Ric = Tic.block<3, 3>(0, 0);
        Eigen::Vector3d tic = Tic.block<3, 1>(0, 3);

        Sophus::Sim3d sophus_Sic(Sophus::RxSO3d(1.0, Ric), tic);
        Sophus::Sim3d sophus_corrected_Siw = sophus_Sic * sophus_sim3_Scw_;
        // Pose corrected with the Sim3 of the loop closure
        sophus_corrected_sim3[pKFi] = sophus_corrected_Siw;
      }

      Eigen::Matrix3d Riw = Tiw.block<3, 3>(0, 0);
      Eigen::Vector3d tiw = Tiw.block<3, 1>(0, 3);
      Sophus::Sim3d sophus_Siw(Sophus::RxSO3d(1.0, Riw), tiw);
      // Pose without correction
      sophus_non_corrected_sim3[pKFi] = sophus_Siw;
    }

    // Correct all MapPoints obsrved by current keyframe and neighbors, so that
    // they align with the other side of the loop
    for (auto mit = sophus_corrected_sim3.begin(),
              mend = sophus_corrected_sim3.end();
         mit != mend; mit++) {
      KeyFrame* pKFi = mit->first;

      Sophus::Sim3d sophus_corrected_Siw = mit->second;
      Sophus::Sim3d sophus_corrected_Swi = sophus_corrected_Siw.inverse();
      Sophus::Sim3d sophus_Siw = sophus_non_corrected_sim3[pKFi];

      std::vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
      for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++) {
        MapPoint* pMPi = vpMPsi[iMP];
        if (!pMPi) {
          continue;
        }
        if (pMPi->isBad()) {
          continue;
        }
        if (pMPi->corrected_by_keyframe_ == current_keyframe_->id_) {
          continue;
        }

        // Project with non-corrected pose and project back with corrected pose
        Eigen::Vector3d P3Dw = pMPi->GetWorldPos();

        Eigen::Matrix<double, 3, 1> corrected_P3Dw = sophus_corrected_Swi * (sophus_Siw * P3Dw);

        pMPi->SetWorldPos(corrected_P3Dw);
        pMPi->corrected_by_keyframe_ = current_keyframe_->id_;
        pMPi->corrected_reference_ = pKFi->id_;
        pMPi->UpdateNormalAndDepth();
      }

      // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3
      // (scale translation)
      Eigen::Matrix3d eigR = sophus_corrected_Siw.rotationMatrix();
      Eigen::Vector3d eigt = sophus_corrected_Siw.translation();
      double s = sophus_corrected_Siw.scale();

      // TODO(b51): check here
      eigt *= (1. / s);  //[R t/s;0 1]
      Eigen::Matrix4d eig_corrected_Tiw = Eigen::Matrix4d::Identity();
      eig_corrected_Tiw.block<3, 3>(0, 0) = eigR;
      eig_corrected_Tiw.block<3, 1>(0, 3) = eigt;

      // Eigen::Matrix4d eig_corrected_Tiw = sophus_corrected_Siw.matrix();
      pKFi->SetPose(eig_corrected_Tiw);

      // Make sure connections are updated
      pKFi->UpdateConnections();
    }

    // Start Loop Fusion
    // Update matched map points and replace if duplicated
    for (size_t i = 0; i < current_matched_map_points_.size(); i++) {
      if (current_matched_map_points_[i]) {
        MapPoint* loop_map_point = current_matched_map_points_[i];
        MapPoint* current_map_point = current_keyframe_->GetMapPoint(i);
        if (current_map_point) {
          current_map_point->Replace(loop_map_point);
        } else {
          current_keyframe_->AddMapPoint(loop_map_point, i);
          loop_map_point->AddObservation(current_keyframe_, i);
          loop_map_point->ComputeDistinctiveDescriptors();
        }
      }
    }
  }

  // Project MapPoints observed in the neighborhood of the loop keyframe
  // into the current keyframe and neighbors using corrected poses.
  // Fuse duplications.
  SearchAndFuse(sophus_corrected_sim3);

  // After the MapPoint fusion, new links in the covisibility graph will appear
  // attaching both sides of the loop
  std::map<KeyFrame*, std::set<KeyFrame*> > loop_connections;

  for (std::vector<KeyFrame*>::iterator
           vit = current_connected_keyframes_.begin(),
           vend = current_connected_keyframes_.end();
       vit != vend; vit++) {
    KeyFrame* pKFi = *vit;
    std::vector<KeyFrame*> vpPreviousNeighbors =
        pKFi->GetVectorCovisibleKeyFrames();

    // Update connections. Detect new links.
    pKFi->UpdateConnections();
    loop_connections[pKFi] = pKFi->GetConnectedKeyFrames();
    for (vector<KeyFrame*>::iterator vit_prev = vpPreviousNeighbors.begin(),
                                     vend_prev = vpPreviousNeighbors.end();
         vit_prev != vend_prev; vit_prev++) {
      loop_connections[pKFi].erase(*vit_prev);
    }
    for (vector<KeyFrame*>::iterator
             vit2 = current_connected_keyframes_.begin(),
             vend2 = current_connected_keyframes_.end();
         vit2 != vend2; vit2++) {
      loop_connections[pKFi].erase(*vit2);
    }
  }

  // Optimize graph
  CeresOptimizer::OptimizeEssentialGraph(
      map_, matched_keyframe_, current_keyframe_, sophus_non_corrected_sim3,
      sophus_corrected_sim3, loop_connections, is_fix_scale_);

  map_->InformNewBigChange();

  // Add loop edge
  matched_keyframe_->AddLoopEdge(current_keyframe_);
  current_keyframe_->AddLoopEdge(matched_keyframe_);

  // Launch a new thread to perform Global Bundle Adjustment
  is_running_global_BA_ = true;
  is_finished_global_BA_ = false;
  is_stop_global_BA_ = false;
  thread_global_BA_ = new thread(&LoopClosing::RunGlobalBundleAdjustment, this,
                                 current_keyframe_->id_);

  // Loop closed. Release Local Mapping.
  local_mapper_->Release();

  last_loop_keyframe_id_ = current_keyframe_->id_;
}

void LoopClosing::SearchAndFuse(const KeyFrameAndSim3& CorrectedPosesMap) {
  ORBmatcher matcher(0.8);

  for (auto mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end();
       mit != mend; mit++) {
    KeyFrame* keyframe = mit->first;

    Sophus::Sim3d Scw = mit->second;
    Eigen::Matrix4d eig_Scw = Scw.matrix();

    vector<MapPoint*> replace_map_points(loop_map_points_.size(),
                                         static_cast<MapPoint*>(nullptr));
    matcher.Fuse(keyframe, eig_Scw, loop_map_points_, 4, replace_map_points);

    // Get Map Mutex
    unique_lock<mutex> lock(map_->mutex_map_update_);
    const int nLP = loop_map_points_.size();
    for (int i = 0; i < nLP; i++) {
      MapPoint* replace_map_point = replace_map_points[i];
      if (replace_map_point) {
        replace_map_point->Replace(loop_map_points_[i]);
      }
    }
  }
}


void LoopClosing::RequestReset() {
  {
    unique_lock<mutex> lock(mutex_reset_);
    is_reset_requested_ = true;
  }

  while (true) {
    {
      unique_lock<mutex> lock2(mutex_reset_);
      if (!is_reset_requested_) {
        break;
      }
    }
    usleep(5000);
  }
}

void LoopClosing::ResetIfRequested() {
  unique_lock<mutex> lock(mutex_reset_);
  if (is_reset_requested_) {
    loop_keyframe_queue_.clear();
    last_loop_keyframe_id_ = 0;
    is_reset_requested_ = false;
  }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF) {
  std::cout << "Starting Global Bundle Adjustment" << endl;

  int index = full_BA_index_;
  CeresOptimizer::GlobalBundleAdjustemnt(map_, 50, &is_stop_global_BA_, nLoopKF,
                                         true);

  // Update all MapPoints and KeyFrames
  // Local Mapping was active during BA, that means that there might be new
  // keyframes not included in the Global BA and they are not consistent with
  // the updated map. We need to propagate the correction through the spanning
  // tree
  {
    unique_lock<mutex> lock(mutex_global_BA_);
    if (index != full_BA_index_) return;

    if (!is_stop_global_BA_) {
      std::cout << "Global Bundle Adjustment finished" << std::endl;
      std::cout << "Updating map ..." << std::endl;
      local_mapper_->RequestStop();
      // Wait until Local Mapping has effectively stopped

      while (!local_mapper_->isStopped() && !local_mapper_->isFinished()) {
        usleep(1000);
      }

      // Get Map Mutex
      unique_lock<mutex> lock(map_->mutex_map_update_);

      // Correct keyframes starting at map first keyframe
      list<KeyFrame*> keyframes_to_check(map_->keyframe_origins_.begin(),
                                         map_->keyframe_origins_.end());

      while (!keyframes_to_check.empty()) {
        KeyFrame* keyframe = keyframes_to_check.front();
        const set<KeyFrame*> childrens = keyframe->GetChilds();
        Eigen::Matrix4d Twc = keyframe->GetPoseInverse();
        for (auto sit = childrens.begin(); sit != childrens.end(); sit++) {
          KeyFrame* child = *sit;
          if (child->n_BA_global_for_keyframe_ != nLoopKF) {
            Eigen::Matrix4d Tchildc = child->GetPose() * Twc;
            child->global_BA_Tcw_ =
                Tchildc * keyframe->global_BA_Tcw_;  //*Tcorc*pKF->mTcwGBA;
            child->n_BA_global_for_keyframe_ = nLoopKF;
          }
          keyframes_to_check.push_back(child);
        }

        keyframe->global_BA_Bef_Tcw_ = keyframe->GetPose();
        keyframe->SetPose(keyframe->global_BA_Tcw_);
        keyframes_to_check.pop_front();
      }

      // Correct MapPoints
      const vector<MapPoint*> map_points = map_->GetAllMapPoints();

      for (size_t i = 0; i < map_points.size(); i++) {
        MapPoint* map_point = map_points[i];

        if (map_point->isBad()) continue;

        if (map_point->n_BA_global_for_keyframe_ == nLoopKF) {
          // If optimized by Global BA, just update
          map_point->SetWorldPos(map_point->global_BA_pose_);
        } else {
          // Update according to the correction of its reference keyframe
          KeyFrame* reference_keyframe = map_point->GetReferenceKeyFrame();

          if (reference_keyframe->n_BA_global_for_keyframe_ != nLoopKF) {
            continue;
          }

          // Map to non-corrected camera
          Eigen::Matrix3d Rcw =
              reference_keyframe->global_BA_Bef_Tcw_.block<3, 3>(0, 0);
          Eigen::Vector3d tcw =
              reference_keyframe->global_BA_Bef_Tcw_.block<3, 1>(0, 3);

          Eigen::Vector3d Xc = Rcw * map_point->GetWorldPos() + tcw;

          // Backproject using corrected camera
          Eigen::Matrix4d Twc = reference_keyframe->GetPoseInverse();
          Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
          Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);

          map_point->SetWorldPos(Rwc * Xc + twc);
        }
      }

      map_->InformNewBigChange();

      local_mapper_->Release();

      cout << "Map updated!" << endl;
    }

    is_finished_global_BA_ = true;
    is_running_global_BA_ = false;
  }
}

void LoopClosing::RequestFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  is_finish_requested_ = true;
}

bool LoopClosing::CheckFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  return is_finish_requested_;
}

void LoopClosing::SetFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  is_finished_ = true;
}

bool LoopClosing::isFinished() {
  unique_lock<mutex> lock(mutex_finish_);
  return is_finished_;
}
}  // namespace ORB_SLAM2
