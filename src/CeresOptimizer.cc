/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: CeresOptimizer.cc
 *
 *          Created On: Tue 10 Sep 2019 03:51:14 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include "CeresOptimizer.h"

#include <ceres/local_parameterization.h>
#include <ceres/solver.h>

#include "Converter.h"

namespace ORB_SLAM2 {
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

  ceres::Problem problem;

  if (keyframes.empty()) {
    LOG(INFO) << "No keyframes, not to optimize";
    return;
  }

  std::map<KeyFrame*, Eigen::Matrix<double, 7, 1> > ided_keyframe_pose;

  for (size_t i = 0; i < keyframes.size(); i++) {
    KeyFrame* keyframe = keyframes[i];

    // Get keyframe Poses
    cv::Mat _kf_pose = keyframe->GetPose();
    Eigen::Matrix<double, 3, 3> keyframe_R;
    Eigen::Matrix<double, 3, 1> keyframe_t;
    keyframe_R << _kf_pose.at<float>(0, 0), _kf_pose.at<float>(0, 1),
        _kf_pose.at<float>(0, 2), _kf_pose.at<float>(1, 0),
        _kf_pose.at<float>(1, 1), _kf_pose.at<float>(1, 2),
        _kf_pose.at<float>(2, 0), _kf_pose.at<float>(2, 1),
        _kf_pose.at<float>(2, 2);
    keyframe_t << _kf_pose.at<float>(0, 3), _kf_pose.at<float>(1, 3),
        _kf_pose.at<float>(2, 3);
    Eigen::Matrix<double, 7, 1> keyframe_Tcw;
    keyframe_Tcw.block<3, 1>(0, 0) = keyframe_t;

    // Eigen Quaternion coeffs output [x, y, z, w]
    keyframe_Tcw.block<4, 1>(3, 0) = Eigen::Quaterniond(keyframe_R).coeffs();

    ided_keyframe_pose[keyframe] = keyframe_Tcw;
  }

  ceres::LossFunction* loss_function = nullptr;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

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
      Eigen::Vector2d observation =
          Eigen::Vector2d(undistort_keypoint.pt.x, undistort_keypoint.pt.y);
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
  options.max_num_iterations = n_iterations;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG(INFO) << summary.FullReport();

  // reset keyframes pose to optimized pose
  for (auto it = ided_keyframe_pose.begin(); it != ided_keyframe_pose.end();
       it++) {
    if (it->first->isBad()) {
      continue;
    }
    Eigen::Matrix<double, 7, 1> Tcw = it->second;
    // Eigen Quaterniond constructed with [w, x, y, z], not same as coeffs
    Eigen::Quaterniond q(Tcw[6], Tcw[3], Tcw[4], Tcw[5]);
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    Eigen::Matrix<double, 4, 4> _pose = Eigen::Matrix<double, 4, 4>::Identity();
    _pose.block<3, 3>(0, 0) = R;
    _pose.block<3, 1>(0, 3) << Tcw[0], Tcw[1], Tcw[2];
    cv::Mat pose = cv::Mat(4, 4, CV_32F);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        pose.at<float>(i, j) = _pose(i, j);
      }
    }
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

void CeresOptimizer::BuildOptimationProblem(
    const std::vector<KeyFrame*>& keyframes,
    const std::vector<MapPoint*>& map_points, ceres::Problem* problem) {
  CHECK(problem != nullptr);
}

}  // namespace ORB_SLAM2
