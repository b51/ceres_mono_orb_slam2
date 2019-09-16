/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: CeresOptimizer.h
 *
 *          Created On: Sat 07 Sep 2019 05:12:19 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef CERES_OPTIMIZER_H_
#define CERES_OPTIMIZER_H_

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <algorithm>

#include "Frame.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"

namespace ORB_SLAM2 {

// Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement. We have two poses x_a
// and x_b. Through sensor measurements we can measure the transformation of
// frame B w.r.t frame A denoted as t_ab_hat. We can compute an error metric
// between the current estimate of the poses and the measurement.
//
// In this formulation, we have chosen to represent the rigid transformation as
// a Hamiltonian quaternion, q, and position, p. The quaternion ordering is
// [x, y, z, w].

// The estimated measurement is:
//       p_cp = [ q_cw * p_wp + p_cw]
//
// measurement transformation. For the orientation error, we will use the
// standard projection error resulting in:
//
//   error = [ undistorted_pixel - K * p_cp / p_cp[2] ]
//
// where Vec(*) returns the vector (imaginary) part of the quaternion. Since
// the measurement has an uncertainty associated with how accurate it is, we
// will weight the errors by the square root of the measurement information
// matrix:
//
//   residuals = I^{1/2) * error
// where I is the information matrix which is the inverse of the covariance.

class PoseGraph3dErrorTerm {
 public:
  PoseGraph3dErrorTerm(const Eigen::Matrix3d& K,
                       const Eigen::Vector2d& observation,
                       const Eigen::Matrix2d& sqrt_information)
      : K_(K), observation_(observation), sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                  const T* const p_b_ptr, T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_keyframe(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_keyframe(q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_point(p_b_ptr);

    // Compute the map point pose in camera frame.
    Eigen::Matrix<T, 3, 1> p_kp = q_keyframe * p_point + p_keyframe;

    // Compute the map point pose in pixel frame.
    Eigen::Matrix<T, 3, 1> projected = K_ * p_kp;

    // Compute the residuals.
    // [ undistorted - projected ]
    Eigen::Map<Eigen::Matrix<T, 2, 1> > residuals(residuals_ptr);
    residuals[0] =
        observation_.template cast<T>()[0] - projected[0] / projected[2];
    residuals[1] =
        observation_.template cast<T>()[1] - projected[1] / projected[2];

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& K,
                                     const Eigen::Vector2d& observation,
                                     const Eigen::Matrix2d& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm,
                                           /* residual numbers */ 2,
                                           /* first optimize numbers */ 3,
                                           /* second optimize numbers */ 4,
                                           /* third optimize numbers */ 3>(
        new PoseGraph3dErrorTerm(K, observation, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The camera intrinsics.
  const Eigen::Matrix3d K_;
  const Eigen::Vector2d observation_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix2d sqrt_information_;
};

class PoseErrorTerm {
 public:
  PoseErrorTerm();
  PoseErrorTerm(const Eigen::Matrix3d& K, const Eigen::Vector2d& observation,
                const Eigen::Vector3d& point_pose,
                const Eigen::Matrix2d& sqrt_information)
      : K_(K),
        observation_(observation),
        point_pose_(point_pose),
        sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_frame(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_frame(q_a_ptr);
    // Compute the map point pose in camera frame.
    Eigen::Matrix<T, 3, 1> p_cp =
        q_frame * point_pose_.template cast<T>() + p_frame;
    // Compute the map point pose in pixel frame.
    Eigen::Matrix<T, 3, 1> projected = K_ * p_cp;

    // Compute the residuals.
    // [ undistorted - projected ]
    Eigen::Map<Eigen::Matrix<T, 2, 1> > residuals(residuals_ptr);
    residuals[0] =
        observation_.template cast<T>()[0] - projected[0] / projected[2];
    residuals[1] =
        observation_.template cast<T>()[1] - projected[1] / projected[2];
    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& K,
                                     const Eigen::Vector2d& observation,
                                     const Eigen::Vector3d& point_pose,
                                     const Eigen::Matrix2d& sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseErrorTerm,
                                           /* residual numbers */ 2,
                                           /* first optimize numbers */ 3,
                                           /* second optimize numbers */ 4>(
        new PoseErrorTerm(K, observation, point_pose, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const Eigen::Matrix3d K_;
  // The point pose in image frame
  const Eigen::Vector2d observation_;
  // The point pose in world frame
  const Eigen::Vector3d point_pose_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix2d sqrt_information_;
};

class CeresOptimizer {
 public:
  void static BundleAdjustment(const std::vector<KeyFrame*>& keyframes,
                               const std::vector<MapPoint*>& map_points,
                               int n_iterations = 200,
                               bool* stop_flag = nullptr,
                               const unsigned long n_loop_keyframe = 0,
                               const bool is_robust = true);

  void static GlobalBundleAdjustemnt(Map* map, int n_iterations = 200,
                                     bool* stop_flag = nullptr,
                                     const unsigned long n_loop_keyframe = 0,
                                     const bool is_robust = true);

  void static LocalBundleAdjustment(KeyFrame* keyframe, bool* stop_flag,
                                    Map* map);

  int static PoseOptimization(Frame* frame);
};

}  // namespace ORB_SLAM2

#endif
