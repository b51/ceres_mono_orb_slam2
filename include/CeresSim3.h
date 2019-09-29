/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: CeresSim3.h
 *
 *          Created On: Mon 23 Sep 2019 09:35:21 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef SIM3_H_
#define SIM3_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

class Sim3 {
 public:
  Sim3()
      : s_(1.),
        q_(Eigen::Quaterniond(1., 0., 0., 0.)),
        t_(Eigen::Vector3d::Zero()) {}

  Sim3(double s, const Eigen::Quaterniond& q, const Eigen::Vector3d& t)
      : s_(s), q_(q), t_(t) {}

  Sim3(double s, const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
      : s_(s), q_(Eigen::Quaterniond(R)), t_(t) {}

  Sim3 inverse() const {
    double s_inverse = 1.0 / s_;
    const Eigen::Quaterniond q_inverse = q_.conjugate();
    const Eigen::Vector3d t_inverse = -(q_inverse * (s_inverse * t_));
    return Sim3(s_inverse, q_inverse, t_inverse);
  }

  inline const double& scale() const { return s_; }

  inline double& scale() { return s_; }

  inline const Eigen::Vector3d& translation() const { return t_; }

  inline Eigen::Vector3d& translation() { return t_; }

  inline const Eigen::Quaterniond& rotation() const { return q_; }

  inline Eigen::Quaterniond& rotation() { return q_; }

  inline Eigen::Vector3d map(const Eigen::Vector3d& xyz) const {
    return s_ * (q_ * xyz) + t_;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  double s_;
  Eigen::Quaterniond q_;
  Eigen::Vector3d t_;
};

Sim3 operator*(const Sim3& lhs, const Sim3& rhs);
inline std::ostream& operator<<(std::ostream& out_str, const Sim3& sim3);
#endif
