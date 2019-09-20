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

#include "Converter.h"

namespace ORB_SLAM2 {

Eigen::Matrix<double, 7, 1> MatToEigen_7_1(cv::Mat pose) {
  Eigen::Matrix<double, 7, 1> Tcw_7_1;
  Eigen::Matrix<double, 3, 3> R;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R(i, j) = pose.at<float>(i, j);
    }
    Tcw_7_1[i] = pose.at<float>(i, 3);
  }
  // Eigen Quaternion coeffs output [x, y, z, w]
  Tcw_7_1.block<4, 1>(3, 0) = Eigen::Quaterniond(R).coeffs();
  return Tcw_7_1;
}

cv::Mat Eigen_7_1_ToMat(Eigen::Matrix<double, 7, 1>& Tcw_7_1) {
  cv::Mat pose = cv::Mat(4, 4, CV_32F);
  Eigen::Quaterniond q(Tcw_7_1[6], Tcw_7_1[3], Tcw_7_1[4], Tcw_7_1[5]);
  Eigen::Matrix3d R = q.normalized().toRotationMatrix();
  Eigen::Matrix<double, 4, 4> _pose = Eigen::Matrix<double, 4, 4>::Identity();
  _pose.block<3, 3>(0, 0) = R;
  _pose.block<3, 1>(0, 3) = Tcw_7_1.block<3, 1>(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      pose.at<float>(i, j) = _pose(i, j);
    }
  }
  return pose.clone();
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

  // ided keyframe for pose recovery after optimize
  std::map<KeyFrame*, Eigen::Matrix<double, 7, 1>> ided_keyframe_pose;

  for (size_t i = 0; i < keyframes.size(); i++) {
    KeyFrame* keyframe = keyframes[i];
    Eigen::Matrix<double, 7, 1> keyframe_Tcw;

    // Get keyframe Poses
    cv::Mat _kf_pose = keyframe->GetPose();
    Eigen::Matrix<double, 3, 3> keyframe_R;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        keyframe_R(i, j) = _kf_pose.at<float>(i, j);
      }
      keyframe_Tcw[i] = _kf_pose.at<float>(i, 3);
    }

    // Eigen Quaternion coeffs output [x, y, z, w]
    keyframe_Tcw.block<4, 1>(3, 0) = Eigen::Quaterniond(keyframe_R).coeffs();

    ided_keyframe_pose[keyframe] = keyframe_Tcw;
  }

  // Setup optimizer
  ceres::Problem problem;
  ceres::LossFunction* loss_function = nullptr;
  if (is_robust) {
    loss_function = new ceres::HuberLoss(sqrt(5.991));
  }
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  // ided map points for pose recovery after optimize
  std::vector<Eigen::Vector3d> ided_map_point_pose(map_points.size(),
                                                   Eigen::Vector3d::Zero());
  ided_map_point_pose.resize(map_points.size());

  for (size_t i = 0; i < map_points.size(); i++) {
    MapPoint* map_point = map_points[i];
    if (map_point->isBad()) {
      continue;
    }

    ided_map_point_pose[i] = Converter::toVector3d(map_point->GetWorldPos());

    const map<KeyFrame*, size_t> observations = map_point->GetObservations();

    int n_edges = 0;
    // Set Edges
    for (auto it = observations.begin(); it != observations.end(); it++) {
      KeyFrame* keyframe = it->first;
      if (keyframe->isBad()) {
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
      problem.SetParameterization(
          ided_keyframe_pose[keyframe].block<4, 1>(3, 0).data(),
          quaternion_local_parameterization);
      // problem->SetParameterization(pose_end_iter->second.q.coeffs().data(),
      //                             quaternion_local_parameterization);
      if (keyframe->id_ == 0) {
        problem.SetParameterBlockConstant(
            ided_keyframe_pose[keyframe].block<3, 1>(0, 0).data());
        problem.SetParameterBlockConstant(
            ided_keyframe_pose[keyframe].block<4, 1>(3, 0).data());
      }
    }
    if (n_edges == 0) {
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
    if (it->first->isBad()) {
      continue;
    }
    Eigen::Matrix<double, 7, 1> Tcw = it->second;
    // Eigen Quaterniond constructed with [w, x, y, z], not same as coeffs
    cv::Mat pose = Eigen_7_1_ToMat(Tcw);
    if (n_loop_keyframe == 0) {
      it->first->SetPose(pose);
    } else {
      it->first->global_BA_Tcw_.create(4, 4, CV_32F);
      pose.copyTo(it->first->global_BA_Tcw_);
      it->first->n_BA_global_for_keyframe_ = n_loop_keyframe;
    }
  }

  // reset map points pose to optimized pose
  for (size_t i = 0; i < ided_map_point_pose.size(); i++) {
    if (is_not_optimized_map_point[i]) {
      continue;
    }
    MapPoint* map_point = map_points[i];

    if (map_point->isBad()) {
      continue;
    }

    cv::Mat pose = cv::Mat(3, 1, CV_32F);
    for (int ii = 0; ii < 3; ii++)
      pose.at<float>(ii) = ided_map_point_pose[i](ii);

    if (n_loop_keyframe == 0) {
      map_point->SetWorldPos(pose);
      map_point->UpdateNormalAndDepth();
    } else {
      map_point->global_BA_pose_.create(3, 1, CV_32F);
      pose.copyTo(map_point->global_BA_pose_);
      map_point->n_BA_global_for_keyframe_ = n_loop_keyframe;
    }
  }
}

bool CeresOptimizer::CheckOutliers(Eigen::Matrix3d K,
                                   Eigen::Vector2d& observation,
                                   Eigen::Vector3d& world_pose,
                                   Eigen::Vector3d& tcw,
                                   Eigen::Quaterniond& qcw, double thres) {
  Eigen::Vector3d pixel_pose = K * (qcw * world_pose + tcw);
  double error_u = observation[0] - pixel_pose[0] / pixel_pose[2];
  double error_v = observation[1] - pixel_pose[1] / pixel_pose[2];
  double error = error_u * error_u + error_v * error_v;
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
      cv::Mat Xw = map_point->GetWorldPos();
      Eigen::Vector3d world_pose(Xw.at<float>(0), Xw.at<float>(1),
                                 Xw.at<float>(2));
      cv::KeyPoint& undistort_keypoint = frame->undistort_keypoints_[i];
      Eigen::Vector2d observation(undistort_keypoint.pt.x,
                                  undistort_keypoint.pt.y);
      double thres = 5.991;

      if (CheckOutliers(K, observation, world_pose, tcw, qcw, thres)) {
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
    cv::Mat _frame_pose = frame->Tcw_.clone();
    Eigen::Matrix<double, 3, 3> frame_R;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        frame_R(i, j) = _frame_pose.at<float>(i, j);
      }
      frame_tcw[i] = _frame_pose.at<float>(i, 3);
    }
    // Eigen Quaternion coeffs output [x, y, z, w]
    frame_qcw = Eigen::Quaterniond(frame_R);

    Eigen::Matrix3d K;
    K << frame->fx_, 0., frame->cx_, 0., frame->fy_, frame->cy_, 0., 0., 1.;

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
        cv::Mat Xw = map_point->GetWorldPos();
        Eigen::Vector3d point_pose(Xw.at<float>(0), Xw.at<float>(1),
                                   Xw.at<float>(2));

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
  Eigen::Matrix<double, 4, 4> _pose = Eigen::Matrix<double, 4, 4>::Identity();
  _pose.block<3, 3>(0, 0) = R;
  _pose.block<3, 1>(0, 3) = frame_tcw;
  cv::Mat pose = cv::Mat(4, 4, CV_32F);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      pose.at<float>(i, j) = _pose(i, j);
    }
  }
  frame->SetPose(pose);
  return n_initial_correspondences - n_bad;
}

void CeresOptimizer::LocalBundleAdjustment(KeyFrame* keyframe, bool* stop_flag,
                                           Map* map) {
  // Local KeyFrames: First Breadth Search from Current Keyframe
  // ided keyframe for pose recovery after optimize
  std::map<KeyFrame*, Eigen::Matrix<double, 7, 1>> ided_local_keyframes;
  ided_local_keyframes[keyframe] = MatToEigen_7_1(keyframe->GetPose());
  keyframe->n_BA_local_for_keyframe_ = keyframe->id_;

  const std::vector<KeyFrame*> neighbor_keyframes =
      keyframe->GetVectorCovisibleKeyFrames();

  for (int i = 0, iend = neighbor_keyframes.size(); i < iend; i++) {
    KeyFrame* neighbor_keyframe = neighbor_keyframes[i];
    neighbor_keyframe->n_BA_local_for_keyframe_ = keyframe->id_;
    if (!neighbor_keyframe->isBad()) {
      ided_local_keyframes[neighbor_keyframe] =
          MatToEigen_7_1(neighbor_keyframe->GetPose());
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
            ided_local_map_points[map_point] =
                Converter::toVector3d(map_point->GetWorldPos());
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
              MatToEigen_7_1(keyframe_i->GetPose());
        }
      }
    }  // for loop end of observations
  }    // for loop end of ided_local_map_points

  // Setup optimizer
  ceres::Problem problem;
  ceres::LossFunction* loss_function = new ceres::HuberLoss(sqrt(5.991));
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (auto it = ided_local_map_points.begin();
       it != ided_local_map_points.end(); it++) {
    MapPoint* map_point = it->first;

    const std::map<KeyFrame*, size_t> observations =
        map_point->GetObservations();

    for (auto it_observation = observations.begin();
         it_observation != observations.end(); it_observation++) {
      KeyFrame* keyframe = it_observation->first;

      if (keyframe->isBad()) continue;

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

      if (ided_local_keyframes.find(keyframe) != ided_local_keyframes.end()) {
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
    }  // end of for loop observations
  }    // end of for loop ided_local_map_points

  ceres::Solver::Options options;
  if (stop_flag) {
    if (*stop_flag) {
      return;
    }
    options.callbacks.push_back(new StopFlagCallback(stop_flag));
  }
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::vector<pair<KeyFrame*, MapPoint*>> to_erase;

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

        bool is_outlier = CheckOutliers(
            K, observation, ided_local_map_points[map_point], tcw, qcw, thres);
        if (is_outlier) {
          to_erase.push_back(std::make_pair(keyframe, map_point));
        }
      }
    }  // end of observations for loop
  }    // end of ided_local_map_points for loop

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
    cv::Mat pose = Eigen_7_1_ToMat(Tcw);
    keyframe->SetPose(pose);
  }
  // MapPoints
  for (auto it = ided_local_map_points.begin();
       it != ided_local_map_points.end(); it++) {
    MapPoint* map_point = it->first;
    map_point->SetWorldPos(Converter::toCvMat(it->second));
    map_point->UpdateNormalAndDepth();
  }
}

}  // namespace ORB_SLAM2
