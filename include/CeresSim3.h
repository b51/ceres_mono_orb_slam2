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
#include <cmath>

template <typename Scalar>
class Sim3 {
 public:
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Quaternion = Eigen::Quaternion<Scalar>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

  using Vector7 = Eigen::Matrix<Scalar, 7, 1>;
  using Matrix7 = Eigen::Matrix<Scalar, 7, 7>;
  Sim3()
      : s_(1.),
        q_(Quaternion(1., 0., 0., 0.)),
        t_(Vector3::Zero()) {}

  Sim3(Scalar s, const Quaternion& q, const Vector3& t)
      : s_(s), q_(q), t_(t) {}

  Sim3(Scalar s, const Matrix3& R, const Vector3& t)
      : s_(s), q_(Quaternion(R)), t_(t) {}

  Sim3 inverse() const {
    Scalar s_inverse = Scalar(1.0) / s_;
    const Quaternion q_inverse = q_.conjugate();
    const Vector3 t_inverse = -(q_inverse * (s_inverse * t_));
    return Sim3(s_inverse, q_inverse, t_inverse);
  }

  static Matrix3 skew(const Vector3& v) {
    Matrix3 m;
    m = Matrix3::Zero();
    m(0, 1) = -v(2);
    m(0, 2) = v(1);
    m(1, 2) = -v(0);
    m(1, 0) = v(2);
    m(2, 0) = -v(1);
    m(2, 1) = v(0);
    return m;
  }

  static Vector3 deltaR(const Matrix3& R) {
    Vector3 v;
    v(0) = R(2, 1) - R(1, 2);
    v(1) = R(0, 2) - R(2, 0);
    v(2) = R(1, 0) - R(0, 1);
    return v;
  }

  static Sim3 exp(const Vector7& update) {
    Vector3 omega;
    for (int i = 0; i < 3; i++) omega[i] = update[i];

    Vector3 upsilon;
    for (int i = 0; i < 3; i++) upsilon[i] = update[i + 3];

    const Scalar& sigma = update[6];
    Scalar theta = omega.norm();
    Matrix3 Omega = skew(omega);
    Scalar s = std::exp(sigma);
    Matrix3 Omega2 = Omega * Omega;
    Matrix3 I;
    I.setIdentity();
    Matrix3 R;

    Scalar eps = Scalar(0.00001);
    Scalar A, B, C;
    if (std::fabs(sigma) < eps) {
      C = Scalar(1.);
      if (theta < eps) {
        A = Scalar(1. / 2.);
        B = Scalar(1. / 6.);
        R = (I + Omega + Omega * Omega);
      } else {
        Scalar theta2 = theta * theta;
        A = (Scalar(1.) - cos(theta)) / (theta2);
        B = (theta - sin(theta)) / (theta2 * theta);
        R = I + sin(theta) / theta * Omega +
            (Scalar(1.) - cos(theta)) / (theta * theta) * Omega2;
      }
    } else {
      C = (s - Scalar(1.)) / sigma;
      if (theta < eps) {
        Scalar sigma2 = sigma * sigma;
        A = ((sigma - Scalar(1.)) * s + Scalar(1.)) / sigma2;
        B = ((Scalar(0.5) * sigma2 - sigma + Scalar(1.)) * s) /
            (sigma2 * sigma);
        R = (I + Omega + Omega2);
      } else {
        R = I + sin(theta) / theta * Omega +
            (Scalar(1.) - cos(theta)) / (theta * theta) * Omega2;

        Scalar a = s * sin(theta);
        Scalar b = s * cos(theta);
        Scalar theta2 = theta * theta;
        Scalar sigma2 = sigma * sigma;

        Scalar c = theta2 + sigma2;
        A = (a * sigma + (Scalar(1.) - b) * theta) / (theta * c);
        B = (C - ((b - Scalar(1.)) * sigma + a * theta) / (c)) * Scalar(1.) /
            (theta2);
      }
    }
    Quaternion q = Quaternion(R);

    Matrix3 W = A * Omega + B * Omega2 + C * I;
    Vector3 t = W * upsilon;
    return Sim3(s, q, t);
  }

  /**
   *  LieGroup to LieAlgebra
   */
  Vector7 log() const {
    Vector7 res;
    const Scalar& sigma = std::log(s_);

    Vector3 omega;
    Vector3 upsilon;

    Matrix3 R = q_.toRotationMatrix();
    Scalar d = Scalar(0.5) * (R(0, 0) + R(1, 1) + R(2, 2) - Scalar(1.));

    Matrix3 Omega;

    Scalar eps = Scalar(0.00001);
    Matrix3 I = Matrix3::Identity();

    Scalar A, B, C;
    if (std::fabs(sigma) < eps) {
      C = Scalar(1.);
      if (d > 1. - eps) {
        omega = 0.5 * deltaR(R);
        Omega = skew(omega);
        A = Scalar(1. / 2.);
        B = Scalar(1. / 6.);
      } else {
        Scalar theta = acos(d);
        Scalar theta2 = theta * theta;
        omega = theta / (2. * sqrt(1. - d * d)) * deltaR(R);
        Omega = skew(omega);
        A = (1. - cos(theta)) / (theta2);
        B = (theta - sin(theta)) / (theta2 * theta);
      }
    } else {
      C = (s_ - 1.) / sigma;
      if (d > 1. - eps) {
        Scalar sigma2 = sigma * sigma;
        omega = 0.5 * deltaR(R);
        Omega = skew(omega);
        A = ((sigma - 1.) * s_ + 1.) / (sigma2);
        B = ((0.5 * sigma2 - sigma + 1.) * s_) / (sigma2 * sigma);
      } else {
        Scalar theta = acos(d);
        omega = theta / (2. * sqrt(1. - d * d)) * deltaR(R);
        Omega = skew(omega);
        Scalar theta2 = theta * theta;
        Scalar a = s_ * sin(theta);
        Scalar b = s_ * cos(theta);
        Scalar c = theta2 + sigma * sigma;
        A = (a * sigma + (1. - b) * theta) / (theta * c);
        B = (C - ((b - 1.) * sigma + a * theta) / (c)) * 1. / (theta2);
      }
    }

    Matrix3 W = A * Omega + B * Omega * Omega + C * I;

    upsilon = W.lu().solve(t_);

    for (int i = 0; i < 3; i++) res[i] = omega[i];

    for (int i = 0; i < 3; i++) res[i + 3] = upsilon[i];

    res[6] = sigma;

    return res;
  }

  inline const Matrix7 adjoint() const {
    const Matrix3& R = q_.toRotationMatrix();
    Matrix7 res;
    res.setZero();
    res.block(0, 0, 3, 3) = s_ * R;
    res.block(0, 3, 3, 3) = skew(t_) * R;
    res.block(0, 6, 3, 1) = -t_;
    res.block(3, 3, 3, 3) = R;
    res(6, 6) = 1;
    return res;
  }

  inline const Scalar& scale() const { return s_; }

  inline Scalar& scale() { return s_; }

  inline const Vector3& translation() const { return t_; }

  inline Vector3& translation() { return t_; }

  inline const Quaternion& rotation() const { return q_; }

  inline Quaternion& rotation() { return q_; }

  inline Vector3 map(const Vector3& xyz) const {
    return s_ * (q_ * xyz) + t_;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  Scalar s_;
  Quaternion q_;
  Vector3 t_;
};

template <typename Scalar>
Sim3<Scalar> operator*(const Sim3<Scalar>& lhs, const Sim3<Scalar>& rhs) {
  return Sim3<Scalar>(
      lhs.scale() * rhs.scale(), lhs.rotation() * rhs.rotation(),
      lhs.scale() * (lhs.rotation() * rhs.translation()) + lhs.translation());
}

template <typename Scalar>
inline std::ostream& operator<<(std::ostream& out_str,
                                const Sim3<Scalar>& sim3) {
  out_str << " [ " << std::endl;
  out_str << "scale: " << sim3.scale() << std::endl;
  out_str << "q.x, q.y, q.z, q.w: " << sim3.rotation().coeffs() << std::endl;
  out_str << "t.x, t.y, t.z: " << sim3.translation() << std::endl;
  out_str << " ] " << std::endl;
  return out_str;
}

using Sim3d = Sim3<double>;
using Sim3f = Sim3<float>;

#endif
