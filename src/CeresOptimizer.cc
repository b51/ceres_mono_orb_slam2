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
  std::vector<bool> is_not_included_map_point;
  is_not_included_map_point.resize(map_points.size());

  ceres::Problem problem;
  BuildOptimationProblem(keyframes, map_points, &problem);

  ceres::Solver::Options options;
  options.max_num_iterations = n_iterations;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG(INFO) << summary.FullReport();
}

void CeresOptimizer::BuildOptimationProblem(
    const std::vector<KeyFrame*>& keyframes,
    const std::vector<MapPoint*>& map_points, ceres::Problem* problem) {
  CHECK(problem != nullptr);

  if (keyframes.empty()) {
    LOG(INFO) << "No keyframes, not to optimize";
    return;
  }

  ceres::LossFunction* loss_function = nullptr;
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (size_t i = 0; i < map_points.size(); i++) {
    MapPoint* map_point = map_points[i];
    if (map_point->isBad()) {
      continue;
    }
    Eigen::Vector3d map_point_pose =
        Converter::toVector3d(map_point->GetWorldPos());
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
      Eigen::Quaterniond keyframe_q(keyframe_R);

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
      problem->AddResidualBlock(cost_function, loss_function, keyframe_t.data(),
                                keyframe_q.coeffs().data(),
                                map_point_pose.data());
      /*
      problem->SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                   quaternion_local_parameterization);
      problem->SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                   quaternion_local_parameterization);
      */
      if (keyframe->id_ == 0) {
        problem->SetParameterBlockConstant(keyframe_t.data());
        problem->SetParameterBlockConstant(keyframe_q.coeffs().data());
      }
    }
  }
}

}  // namespace ORB_SLAM2
