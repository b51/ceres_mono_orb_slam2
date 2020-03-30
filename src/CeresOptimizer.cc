/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: CeresOptimizer.cc
 *
 *          Created On: Tue 10 Sep 2019 03:51:14 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#include "CeresOptimizer.h"

#include <ceres/local_parameterization.h>
#include <ceres/solver.h>
#include <unordered_map>
#include <Eigen/Eigenvalues>

#include "MatEigenConverter.h"
#include <sophus/sim3.hpp>

namespace ORB_SLAM2 {

bool Sim3Parameterization::Plus(const double* x,
                                const double* delta,
                                double* x_plus_delta) const {
  Eigen::Map<const Eigen::Matrix<double, 7, 1> > lie_x(x);
  Eigen::Matrix<double, 7, 1> lie_delta;
  Eigen::Map<Eigen::Matrix<double, 7, 1>> updated(x_plus_delta);

  for (int i = 0; i < 7; i++) {
    lie_delta[i] = delta[i];
  }
  // make sure scale not too small, exp(scale) > 1e-5
  lie_delta[6] = std::max(lie_delta[6], -20.);

  Sophus::Sim3d sim_x = Sophus::Sim3d::exp(lie_x);
  Sophus::Sim3d sim_delta = Sophus::Sim3d::exp(lie_delta);
  updated = (sim_x * sim_delta).log();
  return true;
}

bool Sim3Parameterization::ComputeJacobian(const double* x,
                                           double* jacobian) const {
  ceres::MatrixRef(jacobian, 7, 7) = ceres::Matrix::Identity(7, 7);
  return true;
}

void CeresOptimizer::GlobalBundleAdjustemnt(Map* map, int n_iterations,
                                            bool* stop_flag,
                                            const unsigned long n_loop_keyframe,
                                            const bool is_robust) {
  std::vector<KeyFrame*> keyframes = map->GetAllKeyFrames();
  std::vector<MapPoint*> map_points = map->GetAllMapPoints();
  BundleAdjustment(keyframes, map_points, n_iterations, stop_flag,
                   n_loop_keyframe, is_robust);
}

void CeresOptimizer::BundleAdjustment(const std::vector<KeyFrame*>& keyframes,
                                      const std::vector<MapPoint*>& map_points,
                                      int n_iterations, bool* stop_flag,
                                      const unsigned long n_loop_keyframe,
                                      const bool is_robust) {
  std::vector<bool> is_not_optimized_map_point;
  is_not_optimized_map_point.resize(map_points.size());

  if (keyframes.empty()) {
    LOG(INFO) << "No keyframes, not to optimize";
    return;
  }

  long unsigned int max_keyframe_id = 0;

  // ided keyframe for pose recovery after optimize
  std::map<KeyFrame*, Eigen::Matrix<double, 7, 1>> ided_keyframe_pose;

  // Setup optimizer
  ceres::Problem problem;
  ceres::LossFunction* loss_function = nullptr;
  if (is_robust) {
    loss_function = new ceres::HuberLoss(sqrt(5.991));
  }
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;
  ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;

  for (size_t i = 0; i < keyframes.size(); i++) {
    KeyFrame* keyframe = keyframes[i];
    if (keyframe->isBad()) continue;

    Eigen::Matrix<double, 7, 1> keyframe_Tcw;

    // Get keyframe Poses
    Eigen::Matrix4d kf_pose = keyframe->GetPose();
    Eigen::Matrix3d keyframe_R = kf_pose.block<3, 3>(0, 0);
    keyframe_Tcw.block<3, 1>(0, 0) = kf_pose.block<3, 1>(0, 3);

    // Eigen Quaternion coeffs output [x, y, z, w]
    keyframe_Tcw.block<4, 1>(3, 0) = Eigen::Quaterniond(keyframe_R).coeffs();

    ided_keyframe_pose[keyframe] = keyframe_Tcw;
    if (keyframe->id_ > max_keyframe_id) {
      max_keyframe_id = keyframe->id_;
    }

    problem.AddParameterBlock(
        ided_keyframe_pose[keyframe].block<3, 1>(0, 0).data(), 3);
    problem.AddParameterBlock(
        ided_keyframe_pose[keyframe].block<4, 1>(3, 0).data(), 4,
        quaternion_local_parameterization);
    ordering->AddElementToGroup(
        ided_keyframe_pose[keyframe].block<3, 1>(0, 0).data(), 1);
    ordering->AddElementToGroup(
        ided_keyframe_pose[keyframe].block<4, 1>(3, 0).data(), 1);
    if (keyframe->id_ == 0) {
      problem.SetParameterBlockConstant(
          ided_keyframe_pose[keyframe].block<3, 1>(0, 0).data());
      problem.SetParameterBlockConstant(
          ided_keyframe_pose[keyframe].block<4, 1>(3, 0).data());
    }
  }

  // ided map points for pose recovery after optimize
  std::vector<Eigen::Vector3d> ided_map_point_pose(map_points.size(),
                                                   Eigen::Vector3d::Zero());
  ided_map_point_pose.resize(map_points.size());

  for (size_t i = 0; i < map_points.size(); i++) {
    MapPoint* map_point = map_points[i];
    if (map_point->isBad()) {
      continue;
    }

    ided_map_point_pose[i] = map_point->GetWorldPos();

    const map<KeyFrame*, size_t> observations = map_point->GetObservations();

    int n_edges = 0;
    // Set Edges
    for (auto it = observations.begin(); it != observations.end(); it++) {
      KeyFrame* keyframe = it->first;
      if (keyframe->isBad() || keyframe->id_ > max_keyframe_id) {
        continue;
      }
      n_edges++;

      const cv::KeyPoint& undistort_keypoint =
          keyframe->undistort_keypoints_[it->second];

      Eigen::Matrix3d K;
      K << keyframe->fx_, 0., keyframe->cx_, 0., keyframe->fy_, keyframe->cy_,
          0., 0., 1.;
      Eigen::Vector2d observation(undistort_keypoint.pt.x,
                                  undistort_keypoint.pt.y);
      const float& invSigma2 =
          keyframe->inv_level_sigma2s_[undistort_keypoint.octave];

      Eigen::Matrix2d sqrt_information =
          Eigen::Matrix2d::Identity() * invSigma2;
      ceres::CostFunction* cost_function =
          PoseGraph3dErrorTerm::Create(K, observation, sqrt_information);
      problem.AddResidualBlock(
          cost_function, loss_function,
          ided_keyframe_pose[keyframe].block<3, 1>(0, 0).data(),
          ided_keyframe_pose[keyframe].block<4, 1>(3, 0).data(),
          ided_map_point_pose[i].data());
      ordering->AddElementToGroup(
          ided_map_point_pose[i].data(), 0);
    }
    if (n_edges == 0) {
      problem.RemoveParameterBlock(ided_map_point_pose[i].data());
      is_not_optimized_map_point[i] = true;
    } else {
      is_not_optimized_map_point[i] = false;
    }
  }

  ceres::Solver::Options options;
  if (stop_flag) {
    options.callbacks.push_back(new StopFlagCallback(stop_flag));
  }

  options.max_num_iterations = n_iterations;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // reset keyframes pose to optimized pose
  for (auto it = ided_keyframe_pose.begin(); it != ided_keyframe_pose.end();
       it++) {
    KeyFrame* keyframe = it->first;
    if (keyframe->isBad()) {
      continue;
    }
    Eigen::Matrix<double, 7, 1> Tcw = it->second;
    // Eigen Quaterniond constructed with [w, x, y, z], not same as coeffs
    Eigen::Matrix4d pose = MatEigenConverter::Matrix_7_1_ToMatrix4d(Tcw);
    if (n_loop_keyframe == 0) {
      keyframe->SetPose(pose);
    } else {
      keyframe->global_BA_Tcw_ = pose;
      keyframe->n_BA_global_for_keyframe_ = n_loop_keyframe;
    }
  }

  // reset map points pose to optimized pose
  for (size_t i = 0; i < map_points.size(); i++) {
    if (is_not_optimized_map_point[i]) {
      continue;
    }
    MapPoint* map_point = map_points[i];

    if (map_point->isBad())
      continue;

    if (n_loop_keyframe == 0) {
      map_point->SetWorldPos(ided_map_point_pose[i]);
      map_point->UpdateNormalAndDepth();
    } else {
      map_point->global_BA_pose_ = ided_map_point_pose[i];
      map_point->n_BA_global_for_keyframe_ = n_loop_keyframe;
    }
  }
}

bool CeresOptimizer::CheckOutlier(Eigen::Matrix3d K,
                                  Eigen::Vector2d& observation, float inv_sigma,
                                  Eigen::Vector3d& world_pose,
                                  Eigen::Vector3d& tcw, Eigen::Quaterniond& qcw,
                                  double thres) {
  Eigen::Vector3d pixel_pose = K * (qcw * world_pose + tcw);
  double error_u = observation[0] - pixel_pose[0] / pixel_pose[2];
  double error_v = observation[1] - pixel_pose[1] / pixel_pose[2];
  double error = (error_u * error_u + error_v * error_v) * inv_sigma;
  if (error > thres) {
    return true;
  } else {
    return false;
  }
}

int CeresOptimizer::CheckOutliers(Frame* frame, Eigen::Vector3d& tcw,
                                  Eigen::Quaterniond& qcw) {
  int n_bad = 0;
  Eigen::Matrix3d K;
  K << frame->fx_, 0., frame->cx_, 0., frame->fy_, frame->cy_, 0., 0., 1.;
  const int N = frame->N_;
  for (int i = 0; i < N; i++) {
    MapPoint* map_point = frame->map_points_[i];
    if (map_point) {
      Eigen::Vector3d world_pose = map_point->GetWorldPos();
      cv::KeyPoint& undistort_keypoint = frame->undistort_keypoints_[i];
      Eigen::Vector2d observation(undistort_keypoint.pt.x,
                                  undistort_keypoint.pt.y);
      float inv_sigma = frame->inv_level_sigma2s_[undistort_keypoint.octave];
      double thres = 5.991;

      if (CheckOutlier(K, observation, inv_sigma, world_pose, tcw, qcw,
                       thres)) {
        frame->is_outliers_[i] = true;
        n_bad++;
      } else {
        frame->is_outliers_[i] = false;
      }
    }
  }
  return n_bad;
}

// Optimize frame pose with poeses of map points that matched with last frame,
// use projection to calculate residuals for Tcw optimizing
// residuals = undistorted_point - K * [Tcw * Pwp]

int CeresOptimizer::PoseOptimization(Frame* frame) {
  ceres::Problem problem;
  const int N = frame->N_;
  int n_initial_correspondences = 0;

  Eigen::Vector3d frame_tcw;
  Eigen::Quaterniond frame_qcw;
  int n_bad = 0;
  {
    unique_lock<mutex> lock(MapPoint::global_mutex_);
    // Get frame Pose
    Eigen::Matrix4d frame_pose = frame->Tcw_;
    Eigen::Matrix3d frame_R;
    frame_R = frame_pose.block<3, 3>(0, 0);
    frame_tcw = frame_pose.block<3, 1>(0, 3);
    // Eigen Quaternion coeffs output [x, y, z, w]
    frame_qcw = Eigen::Quaterniond(frame_R);

    Eigen::Matrix3d K = MatEigenConverter::MatToMatrix3d(frame->K_);

    // ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.5);
    ceres::LossFunction* loss_function = new ceres::HuberLoss(sqrt(5.991));
    ceres::LocalParameterization* quaternion_local_parameterization =
        new ceres::EigenQuaternionParameterization;
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;

    n_initial_correspondences = 0;
    for (int i = 0; i < N; i++) {
      MapPoint* map_point = frame->map_points_[i];
      if (map_point) {
        n_initial_correspondences++;
        frame->is_outliers_[i] = false;
        // Monocular observation
        Eigen::Vector3d point_pose = map_point->GetWorldPos();

        const cv::KeyPoint& undistort_keypoint = frame->undistort_keypoints_[i];
        Eigen::Vector2d observation(undistort_keypoint.pt.x,
                                    undistort_keypoint.pt.y);

        const float invSigma2 =
            frame->inv_level_sigma2s_[undistort_keypoint.octave];
        Eigen::Matrix2d sqrt_information =
            Eigen::Matrix2d::Identity() * invSigma2;

        ceres::CostFunction* cost_function =
            PoseErrorTerm::Create(K, observation, point_pose, sqrt_information);
        problem.AddResidualBlock(cost_function, loss_function, frame_tcw.data(),
                                 frame_qcw.coeffs().data());
        problem.SetParameterization(frame_qcw.coeffs().data(),
                                    quaternion_local_parameterization);
      }
    }
    if (n_initial_correspondences < 3) return 0;

    ceres::Solve(options, &problem, &summary);
    n_bad = CheckOutliers(frame, frame_tcw, frame_qcw);
  }

  Eigen::Matrix3d R = frame_qcw.normalized().toRotationMatrix();
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3, 3>(0, 0) = R;
  pose.block<3, 1>(0, 3) = frame_tcw;
  frame->SetPose(pose);
  return n_initial_correspondences - n_bad;
}

void CeresOptimizer::LocalBundleAdjustment(KeyFrame* keyframe, bool* stop_flag,
                                           Map* map) {
  // Local KeyFrames: First Breadth Search from Current Keyframe
  // ided keyframe for pose recovery after optimize
  std::unordered_map<KeyFrame*, Eigen::Matrix<double, 7, 1>> ided_local_keyframes;
  ided_local_keyframes[keyframe] =
      MatEigenConverter::Matrix4dToMatrix_7_1(keyframe->GetPose());
  keyframe->n_BA_local_for_keyframe_ = keyframe->id_;

  const std::vector<KeyFrame*> neighbor_keyframes =
      keyframe->GetVectorCovisibleKeyFrames();

  for (int i = 0, iend = neighbor_keyframes.size(); i < iend; i++) {
    KeyFrame* neighbor_keyframe = neighbor_keyframes[i];
    neighbor_keyframe->n_BA_local_for_keyframe_ = keyframe->id_;
    if (!neighbor_keyframe->isBad()) {
      ided_local_keyframes[neighbor_keyframe] =
          MatEigenConverter::Matrix4dToMatrix_7_1(neighbor_keyframe->GetPose());
    }
  }

  // Local MapPoints seen in Local KeyFrames
  std::map<MapPoint*, Eigen::Vector3d> ided_local_map_points;

  for (auto it = ided_local_keyframes.begin(); it != ided_local_keyframes.end();
       it++) {
    std::vector<MapPoint*> map_points = it->first->GetMapPointMatches();

    for (auto it_map_point = map_points.begin();
         it_map_point != map_points.end(); it_map_point++) {
      MapPoint* map_point = *it_map_point;
      if (map_point) {
        if (!map_point->isBad()) {
          if (map_point->n_BA_local_for_keyframe_ != keyframe->id_) {
            ided_local_map_points[map_point] = map_point->GetWorldPos();
            map_point->n_BA_local_for_keyframe_ = keyframe->id_;
          }
        }
      }
    }  // for loop end of map_points
  }    // for loop end of ided_local_keyframes

  // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local
  // Keyframes
  std::map<KeyFrame*, Eigen::Matrix<double, 7, 1>> ided_fixed_keyframes;
  for (auto it = ided_local_map_points.begin();
       it != ided_local_map_points.end(); it++) {
    std::map<KeyFrame*, size_t> observations = it->first->GetObservations();

    for (auto it_observation = observations.begin();
         it_observation != observations.end(); it_observation++) {
      KeyFrame* keyframe_i = it_observation->first;

      if (keyframe_i->n_BA_local_for_keyframe_ != keyframe->id_ &&
          keyframe_i->n_BA_fixed_for_keyframe_ != keyframe->id_) {
        keyframe_i->n_BA_fixed_for_keyframe_ = keyframe->id_;
        if (!keyframe_i->isBad()) {
          ided_fixed_keyframes[keyframe_i] =
              MatEigenConverter::Matrix4dToMatrix_7_1(keyframe_i->GetPose());
        }
      }
    }  // for loop end of observations
  }    // for loop end of ided_local_map_points

  // Setup optimizer
  ceres::Problem problem;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  int optimize_count = 0;
  std::vector<std::pair<KeyFrame*, MapPoint*>> to_erase;

reoptimize:
  ceres::ParameterBlockOrdering* ordering = nullptr;
  ordering = new ceres::ParameterBlockOrdering;
  ceres::LossFunction* loss_function = nullptr;
  if (optimize_count == 0)
    loss_function = new ceres::HuberLoss(sqrt(5.991));

  for (auto it = ided_local_map_points.begin();
       it != ided_local_map_points.end(); it++) {
    MapPoint* map_point = it->first;
    const std::map<KeyFrame*, size_t> observations =
        map_point->GetObservations();

    for (auto it_observation = observations.begin();
         it_observation != observations.end(); it_observation++) {
      KeyFrame* keyframe = it_observation->first;
      if (keyframe->isBad()) continue;

      if (!to_erase.empty() and
          std::find(to_erase.begin(), to_erase.end(),
                    std::make_pair(keyframe, map_point)) != to_erase.end()) {
        continue;
      }

      // Get values
      const cv::KeyPoint& undistort_keypoint =
          keyframe->undistort_keypoints_[it_observation->second];
      Eigen::Matrix3d K;
      K << keyframe->fx_, 0., keyframe->cx_, 0., keyframe->fy_, keyframe->cy_,
          0., 0., 1.;
      Eigen::Vector2d observation(undistort_keypoint.pt.x,
                                  undistort_keypoint.pt.y);
      const float& invSigma2 =
          keyframe->inv_level_sigma2s_[undistort_keypoint.octave];
      Eigen::Matrix2d sqrt_information =
          Eigen::Matrix2d::Identity() * invSigma2;

      // create cost function
      ceres::CostFunction* cost_function =
          PoseGraph3dErrorTerm::Create(K, observation, sqrt_information);

      ordering->AddElementToGroup(it->second.data(), 0);
      if (ided_local_keyframes.find(keyframe) != ided_local_keyframes.end()) {
        // points ordering 1
        ordering->AddElementToGroup(
            ided_local_keyframes[keyframe].block<3, 1>(0, 0).data(), 1);
        ordering->AddElementToGroup(
            ided_local_keyframes[keyframe].block<4, 1>(3, 0).data(), 1);
        // add residual block
        problem.AddResidualBlock(
            cost_function, loss_function,
            ided_local_keyframes[keyframe].block<3, 1>(0, 0).data(),
            ided_local_keyframes[keyframe].block<4, 1>(3, 0).data(),
            it->second.data());
        // set q to quaternion parameterization
        problem.SetParameterization(
            ided_local_keyframes[keyframe].block<4, 1>(3, 0).data(),
            quaternion_local_parameterization);

        // must set first keyframe to constant
        if (keyframe->id_ == 0) {
          problem.SetParameterBlockConstant(
              ided_local_keyframes[keyframe].block<3, 1>(0, 0).data());
          problem.SetParameterBlockConstant(
              ided_local_keyframes[keyframe].block<4, 1>(3, 0).data());
        }
      } else {
        if (ided_fixed_keyframes.find(keyframe) != ided_fixed_keyframes.end()) {
          // add points ordering 1
          ordering->AddElementToGroup(
              ided_fixed_keyframes[keyframe].block<3, 1>(0, 0).data(), 1);
          ordering->AddElementToGroup(
              ided_fixed_keyframes[keyframe].block<4, 1>(3, 0).data(), 1);
          // add residual block
          problem.AddResidualBlock(
              cost_function, loss_function,
              ided_fixed_keyframes[keyframe].block<3, 1>(0, 0).data(),
              ided_fixed_keyframes[keyframe].block<4, 1>(3, 0).data(),
              it->second.data());
          // set q to quaternion parameterization
          problem.SetParameterization(
              ided_fixed_keyframes[keyframe].block<4, 1>(3, 0).data(),
              quaternion_local_parameterization);
          problem.SetParameterBlockConstant(
              ided_fixed_keyframes[keyframe].block<3, 1>(0, 0).data());
          problem.SetParameterBlockConstant(
              ided_fixed_keyframes[keyframe].block<4, 1>(3, 0).data());
        }
      }
    }  // end of for loop observations
  }    // end of for loop ided_local_map_points

  ceres::Solver::Options options;
  if (stop_flag) {
    if (*stop_flag) {
      return;
    }
    options.callbacks.push_back(new StopFlagCallback(stop_flag));
  }
  options.linear_solver_ordering.reset(ordering);
  options.num_threads = 4;
  options.max_num_iterations = 10;
  if (optimize_count == 0)
    options.max_num_iterations = 5;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.use_explicit_schur_complement = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::vector<std::pair<KeyFrame*, MapPoint*>> empty_to_erase;
  to_erase.swap(empty_to_erase);

  for (auto it = ided_local_map_points.begin();
       it != ided_local_map_points.end(); it++) {
    MapPoint* map_point = it->first;
    if (map_point->isBad()) continue;

    auto observations = map_point->GetObservations();

    for (auto it_observation = observations.begin();
         it_observation != observations.end(); it_observation++) {
      // get map point [u, v] in keyframe pixel coordinate
      KeyFrame* keyframe = it_observation->first;
      const cv::KeyPoint& undistort_keypoint =
          keyframe->undistort_keypoints_[it_observation->second];
      Eigen::Vector2d observation(undistort_keypoint.pt.x,
                                  undistort_keypoint.pt.y);
      float inv_sigma = keyframe->inv_level_sigma2s_[undistort_keypoint.octave];

      // check is map point outlier in optimized keyframes
      if (ided_local_keyframes.find(keyframe) != ided_local_keyframes.end()) {
        Eigen::Vector3d tcw = ided_local_keyframes[keyframe].block<3, 1>(0, 0);
        Eigen::Vector4d q_xyzw =
            ided_local_keyframes[keyframe].block<4, 1>(3, 0);
        Eigen::Quaterniond qcw(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);

        double thres = 5.991;
        Eigen::Matrix3d K;
        K << keyframe->fx_, 0., keyframe->cx_, 0., keyframe->fy_, keyframe->cy_,
            0., 0., 1.;

        bool is_outlier =
            CheckOutlier(K, observation, inv_sigma,
                         ided_local_map_points[map_point], tcw, qcw, thres);
        Eigen::Vector3d map_point_c = qcw * ided_local_map_points[map_point] + tcw;
        if (is_outlier || map_point_c[2] <= 0) {
          to_erase.push_back(std::make_pair(keyframe, map_point));
        }
      }
    }  // end of observations for loop
  }    // end of ided_local_map_points for loop
  if (optimize_count == 0) {
    optimize_count++;
    goto reoptimize;
  }

  unique_lock<mutex> lock(map->mutex_map_update_);
  if (!to_erase.empty()) {
    for (size_t i = 0; i < to_erase.size(); i++) {
      KeyFrame* keyframe = to_erase[i].first;
      MapPoint* map_point = to_erase[i].second;
      keyframe->EraseMapPointMatch(map_point);
      map_point->EraseObservation(keyframe);
    }
  }

  // Recover optimized data
  // Keyframes
  for (auto it = ided_local_keyframes.begin(); it != ided_local_keyframes.end();
       it++) {
    KeyFrame* keyframe = it->first;
    Eigen::Matrix<double, 7, 1> Tcw = it->second;
    Eigen::Matrix4d pose = MatEigenConverter::Matrix_7_1_ToMatrix4d(Tcw);
    keyframe->SetPose(pose);
  }
  // MapPoints
  for (auto it = ided_local_map_points.begin();
       it != ided_local_map_points.end(); it++) {
    MapPoint* map_point = it->first;
    map_point->SetWorldPos(it->second);
    map_point->UpdateNormalAndDepth();
  }
}

int CeresOptimizer::OptimizeSim3(KeyFrame* keyframe_1, KeyFrame* keyframe_2,
                                 std::vector<MapPoint*>& matches12,
                                 Sophus::Sim3d& S12, const float th2,
                                 const bool bFixScale) {
  Eigen::Matrix<double, 7, 1> sim12 = S12.log();

  Eigen::Matrix3d K1 = MatEigenConverter::MatToMatrix3d(keyframe_1->K_);
  Eigen::Matrix3d K2 = MatEigenConverter::MatToMatrix3d(keyframe_2->K_);

  // Camera poses
  Eigen::Matrix3d R1cw = keyframe_1->GetRotation();
  Eigen::Vector3d t1cw = keyframe_1->GetTranslation();

  Eigen::Matrix3d R2cw = keyframe_2->GetRotation();
  Eigen::Vector3d t2cw = keyframe_2->GetTranslation();

  const int N = matches12.size();
  const std::vector<MapPoint*> map_points_1 = keyframe_1->GetMapPointMatches();

  int n_correspondences = 0;

  const double deltaHuber = sqrt(th2);
  ceres::Problem problem;
  ceres::LossFunction* loss_function = new ceres::HuberLoss(deltaHuber);
  ceres::LocalParameterization* sim3_local_parameterization =
      new Sim3Parameterization;
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  ceres::Solver::Summary summary;

  // for CheckOutliers
  std::list<Eigen::Vector2d> obs1_list;
  std::list<Eigen::Vector3d> P3D2c_list;
  std::list<Eigen::Vector2d> obs2_list;
  std::list<Eigen::Vector3d> P3D1c_list;
  std::list<float> inv_sigma_1_list;
  std::list<float> inv_sigma_2_list;

  for (int i = 0; i < N; i++) {
    if (!matches12[i]) continue;

    MapPoint* map_point_1 = map_points_1[i];
    MapPoint* map_point_2 = matches12[i];

    const int i2 = map_point_2->GetIndexInKeyFrame(keyframe_2);

    if (map_point_1 && map_point_2) {
      if (!map_point_1->isBad() && !map_point_2->isBad() && i2 >= 0) {
        n_correspondences++;

        // Get observations in keyframe_1's pixel coordinate
        const cv::KeyPoint& keypoint_1 = keyframe_1->undistort_keypoints_[i];
        Eigen::Vector2d obs1(keypoint_1.pt.x, keypoint_1.pt.y);
        float inv_sigma_1 = keyframe_1->inv_level_sigma2s_[keypoint_1.octave];
        Eigen::Matrix2d invSigmaSquareInfo1 =
            inv_sigma_1 * Eigen::Matrix2d::Identity();
        // Project map point in keyframe_2 to keyframe_1
        Eigen::Vector3d P3D2w = map_point_2->GetWorldPos();
        Eigen::Vector3d P3D2c = R2cw * P3D2w + t2cw;
        // Create residual function, residual = obs1 - K1 * (sT12 * P3D2c),
        // sT12 -> do_inverse = false;
        ceres::CostFunction* obs1_cost_function =
            Sim3ErrorTerm::Create(K1, obs1, P3D2c, invSigmaSquareInfo1, false);
        problem.AddResidualBlock(obs1_cost_function, loss_function,
                                 sim12.data());
        problem.SetParameterization(sim12.data(), sim3_local_parameterization);

        // Get observations in keyframe_2's pixel coordinate
        const cv::KeyPoint& keypoint_2 = keyframe_2->undistort_keypoints_[i2];
        Eigen::Vector2d obs2(keypoint_2.pt.x, keypoint_2.pt.y);
        float inv_sigma_2 = keyframe_2->inv_level_sigma2s_[keypoint_2.octave];
        Eigen::Matrix2d invSigmaSquareInfo2 =
            inv_sigma_2 * Eigen::Matrix2d::Identity();
        // Project map point in keyframe_1 to keyframe_2
        Eigen::Vector3d P3D1w = map_point_1->GetWorldPos();
        Eigen::Vector3d P3D1c = R1cw * P3D1w + t1cw;
        // Residual function, residual = obs2 - K2 * (sT12.inverse() * P3D1c)
        // sT21 = sT12.inverse() -> do_inverse = true;
        ceres::CostFunction* obs2_cost_function =
            Sim3ErrorTerm::Create(K2, obs2, P3D1c, invSigmaSquareInfo2, true);
        problem.AddResidualBlock(obs2_cost_function, loss_function,
                                 sim12.data());
        problem.SetParameterization(sim12.data(), sim3_local_parameterization);
        // push for outliers count
        obs1_list.emplace_back(obs1);
        inv_sigma_1_list.emplace_back(inv_sigma_1);
        P3D2c_list.emplace_back(P3D2c);
        obs2_list.emplace_back(obs2);
        inv_sigma_2_list.emplace_back(inv_sigma_2);
        P3D1c_list.emplace_back(P3D1c);
      }
    }
  }  // end of for N loop

  ceres::Solve(options, &problem, &summary);

  S12 = Sophus::Sim3d::exp(sim12);

  int n_bad = 0;

  while (!obs1_list.empty()) {
    Eigen::Matrix3d sR12_for_check = S12.scale() * S12.rotationMatrix();
    Eigen::Vector3d t12_for_check = S12.translation();
    Eigen::Quaterniond q12_for_check(sR12_for_check);

    bool is_outlier_12 = CheckOutlier(
        K1, obs1_list.front(), inv_sigma_1_list.front(), P3D2c_list.front(),
        t12_for_check, q12_for_check, deltaHuber * deltaHuber);

    Sophus::Sim3d S21 = S12.inverse();
    Eigen::Matrix3d sR21_for_check = S21.scale() * S21.rotationMatrix();
    Eigen::Vector3d t21_for_check = S21.translation();
    Eigen::Quaterniond q21_for_check(sR21_for_check);

    bool is_outlier_21 = CheckOutlier(
        K2, obs2_list.front(), inv_sigma_2_list.front(), P3D1c_list.front(),
        t21_for_check, q21_for_check, deltaHuber * deltaHuber);

    obs1_list.pop_front();
    inv_sigma_1_list.pop_front();
    P3D2c_list.pop_front();
    obs2_list.pop_front();
    inv_sigma_2_list.pop_front();
    P3D1c_list.pop_front();

    if (is_outlier_12 or is_outlier_21) n_bad++;
  }

  VLOG(4) << "n_correspondences: " << n_correspondences << " n_bad: " << n_bad;

  if (n_correspondences - n_bad < 10) return 0;

  return n_correspondences - n_bad;
}

void CeresOptimizer::OptimizeEssentialGraph(
    Map* map, KeyFrame* loop_keyframe, KeyFrame* current_keyframe,
    const LoopClosing::KeyFrameAndSim3& keyframes_non_corrected_sim3,
    const LoopClosing::KeyFrameAndSim3& keyframes_corrected_sim3,
    const std::map<KeyFrame*, std::set<KeyFrame*>>& loop_connections,
    const bool& is_fixed_scale) {
  ceres::Problem problem;
  ceres::LocalParameterization* sim3_local_parameterization =
      new Sim3Parameterization;

  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;

  int min_weight = 100;
  const Eigen::Matrix<double, 7, 7> sqrt_information =
      Eigen::Matrix<double, 7, 7>::Identity();

  const std::vector<KeyFrame*> all_keyframes = map->GetAllKeyFrames();
  const std::vector<MapPoint*> all_map_points = map->GetAllMapPoints();

  const unsigned int max_keyframe_id = map->GetMaxKFid();

  std::vector<Sophus::Sim3d, Eigen::aligned_allocator<Sophus::Sim3d>>
      corrected_Swcs(max_keyframe_id + 1);

  std::vector<Eigen::Matrix<double, 7, 1>> Scw_datas(max_keyframe_id + 1);
  std::vector<Eigen::Matrix<double, 7, 1>> Scw_original_datas(max_keyframe_id + 1);

  // Set all keyframe
  for (size_t i = 0; i < all_keyframes.size(); i++) {
    KeyFrame* keyframe = all_keyframes[i];
    if (keyframe->isBad()) continue;

    const int id_i = keyframe->id_;

    auto it = keyframes_corrected_sim3.find(keyframe);
    if (it != keyframes_corrected_sim3.end()) {
      Scw_datas[id_i] = it->second.log();
    } else {
      Eigen::Matrix3d Rcw = keyframe->GetRotation();
      Eigen::Vector3d tcw = keyframe->GetTranslation();
      Sophus::Sim3d Siw(Sophus::RxSO3d(1.0, Rcw), tcw);
      Scw_datas[id_i] = Siw.log();
    }
    Scw_original_datas[id_i] = Scw_datas[id_i];

    problem.AddParameterBlock(Scw_datas[id_i].data(), 7,
                              sim3_local_parameterization);
    if (keyframe == loop_keyframe) {
      // LOG(INFO) << "constant keyframe: " << keyframe->id_;
      problem.SetParameterBlockConstant(Scw_datas[id_i].data());
    }
  }  // end of all_keyframes loop

  std::set<std::pair<long unsigned int, long unsigned int>> inserted_edges;

  // Set Loop connections
  for (auto it = loop_connections.begin(); it != loop_connections.end(); it++) {
    KeyFrame* keyframe = it->first;
    const long unsigned int id_i = keyframe->id_;
    const std::set<KeyFrame*>& connected_keyframes = it->second;
    const Sophus::Sim3d Siw = Sophus::Sim3d::exp(Scw_datas[id_i]);
    const Sophus::Sim3d Swi = Siw.inverse();

    for (auto it_connections = connected_keyframes.begin();
         it_connections != connected_keyframes.end(); it_connections++) {
      const long unsigned int id_j = (*it_connections)->id_;
      if ((id_i != current_keyframe->id_ || id_j != loop_keyframe->id_) &&
          keyframe->GetWeight(*it_connections) < min_weight)
        continue;

      const Sophus::Sim3d Sjw = Sophus::Sim3d::exp(Scw_datas[id_j]);
      const Sophus::Sim3d Sji = Sjw * Swi;

      ceres::CostFunction* cost_function =
          EssentialGraphErrorTerm::Create(Sji, sqrt_information);
      problem.AddResidualBlock(cost_function, nullptr, Scw_datas[id_j].data(),
                               Scw_datas[id_i].data());

      inserted_edges.insert(std::make_pair(std::min(id_i, id_j), std::max(id_i, id_j)));
    }
  }  // end of loop connections

  // Set normal keyframes
  for (size_t i = 0; i < all_keyframes.size(); i++) {
    KeyFrame* keyframe = all_keyframes[i];

    const int id_i = keyframe->id_;
    Sophus::Sim3d Swi;

    auto it_i = keyframes_non_corrected_sim3.find(keyframe);
    if (it_i != keyframes_non_corrected_sim3.end()) {
      Swi = (it_i->second).inverse();
    } else {
      Swi = Sophus::Sim3d::exp(Scw_datas[id_i]).inverse();
    }
    KeyFrame* parent_keyframe = keyframe->GetParent();
    if (parent_keyframe) {
      long unsigned int id_j = parent_keyframe->id_;

      Sophus::Sim3d Sjw;

      auto it_j = keyframes_non_corrected_sim3.find(parent_keyframe);
      if (it_j != keyframes_non_corrected_sim3.end()) {
        Sjw = it_j->second;
      } else {
        Sjw = Sophus::Sim3d::exp(Scw_datas[id_j]);
      }
      Sophus::Sim3d Sji = Sjw * Swi;

      ceres::CostFunction* cost_function =
          EssentialGraphErrorTerm::Create(Sji, sqrt_information);
      problem.AddResidualBlock(cost_function, nullptr, Scw_datas[id_j].data(),
                               Scw_datas[id_i].data());
    }

    // Set for loop edges
    const std::set<KeyFrame*> loop_edges = keyframe->GetLoopEdges();
    for (auto it = loop_edges.begin(); it != loop_edges.end(); it++) {
      KeyFrame* local_loop_keyframe = *it;
      if (local_loop_keyframe->id_ < keyframe->id_) {
        Sophus::Sim3d Slw;
        auto it_l = keyframes_non_corrected_sim3.find(local_loop_keyframe);
        if (it_l != keyframes_non_corrected_sim3.end()) {
          Slw = it_l->second;
        } else {
          Slw = Sophus::Sim3d::exp(Scw_datas[local_loop_keyframe->id_]);
        }
        Sophus::Sim3d Sli = Slw * Swi;

        ceres::CostFunction* cost_function =
            EssentialGraphErrorTerm::Create(Sli, sqrt_information);
        problem.AddResidualBlock(cost_function, nullptr,
                                 Scw_datas[local_loop_keyframe->id_].data(),
                                 Scw_datas[id_i].data());
      }
    }

    // Covisibility keyframes
    const std::vector<KeyFrame*> connected_keyframes =
        keyframe->GetCovisiblesByWeight(min_weight);
    for (auto it = connected_keyframes.begin(); it != connected_keyframes.end();
         it++) {
      KeyFrame* keyframe_n = *it;
      if (keyframe_n && keyframe_n != parent_keyframe &&
          !keyframe->hasChild(keyframe_n) && !loop_edges.count(keyframe_n)) {
        if (!keyframe_n->isBad() && keyframe_n->id_ < keyframe->id_) {
          if (inserted_edges.count(
                  std::make_pair(min(keyframe->id_, keyframe_n->id_),
                                 max(keyframe->id_, keyframe_n->id_))))
            continue;

          Sophus::Sim3d Snw;
          auto it_n = keyframes_non_corrected_sim3.find(keyframe_n);
          if (it_n != keyframes_non_corrected_sim3.end())
            Snw = it_n->second;
          else
            Snw = Sophus::Sim3d::exp(Scw_datas[keyframe_n->id_]);

          Sophus::Sim3d Sni = Snw * Swi;

          ceres::CostFunction* cost_function =
              EssentialGraphErrorTerm::Create(Sni, sqrt_information);
          problem.AddResidualBlock(cost_function, nullptr,
                                   Scw_datas[keyframe_n->id_].data(),
                                   Scw_datas[id_i].data());
        }
      }
    }
  }  // end of for loop all_keyframes

  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << summary.FullReport();

  unique_lock<mutex> lock(map->mutex_map_update_);

  // SE3 Pose Recovering. Sim3d:[sR t;0 1] -> SE3:[R t/s;0 1]
  for (size_t i = 0; i < all_keyframes.size(); i++) {
    KeyFrame* keyframe = all_keyframes[i];
    const int id_i = keyframe->id_;

    Sophus::Sim3d sim3_corrected_Scw = Sophus::Sim3d::exp(Scw_datas[id_i]);
    corrected_Swcs[id_i] = sim3_corrected_Scw.inverse();
    Eigen::Matrix3d R = sim3_corrected_Scw.rotationMatrix();
    Eigen::Vector3d t = sim3_corrected_Scw.translation();
    double s = sim3_corrected_Scw.scale();

    Eigen::Matrix4d Tiw = Eigen::Matrix4d::Identity();
    t = (1. / s) * t;
    Tiw.block<3, 3>(0, 0) = R;
    Tiw.block<3, 1>(0, 3) = t;
    // Eigen::Matrix4d Tiw = sim3_corrected_Scw.matrix();
    keyframe->SetPose(Tiw);
  }
  // Correct points. Transform to "non-optimized" reference keyframe pose and
  // transform back with optimized pose
  for (size_t i = 0; i < all_map_points.size(); i++) {
    MapPoint* map_point = all_map_points[i];

    if (map_point->isBad()) continue;

    int id_r;
    if (map_point->corrected_by_keyframe_ == current_keyframe->id_) {
      id_r = map_point->corrected_reference_;
    } else {
      KeyFrame* reference_keyframe = map_point->GetReferenceKeyFrame();
      id_r = reference_keyframe->id_;
    }
    Sophus::Sim3d Srw = Sophus::Sim3d::exp(Scw_original_datas[id_r]);
    Sophus::Sim3d corrected_Swr = corrected_Swcs[id_r];
    Eigen::Vector3d P3Dw = map_point->GetWorldPos();
    Eigen::Vector3d P3Dc = Srw * P3Dw;
    Eigen::Vector3d corrected_P3Dw = corrected_Swr * P3Dc;

    map_point->SetWorldPos(corrected_P3Dw);
    map_point->UpdateNormalAndDepth();
  }
}

}  // namespace ORB_SLAM2
