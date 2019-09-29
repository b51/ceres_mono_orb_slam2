/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: MatEigenConverter.cc
 *
 *          Created On: Fri 20 Sep 2019 06:22:54 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#include "MatEigenConverter.h"

Eigen::Matrix4d MatEigenConverter::MatToMatrix4d(const cv::Mat& T) {
  Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Zero();
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      eigen_T(i, j) = T.at<float>(i, j);
    }
  return eigen_T;
}

cv::Mat MatEigenConverter::Matrix4dToMat(const Eigen::Matrix4d& T) {
  cv::Mat cv_T = cv::Mat(4, 4, CV_32F);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      cv_T.at<float>(i, j) = T(i, j);
    }
  return cv_T.clone();
}

Eigen::Matrix3d MatEigenConverter::MatToMatrix3d(const cv::Mat& R) {
  Eigen::Matrix3d eigen_R = Eigen::Matrix3d::Zero();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      eigen_R(i, j) = R.at<float>(i, j);
    }
  return eigen_R;
}

cv::Mat MatEigenConverter::Matrix3dToMat(const Eigen::Matrix3d& R) {
  cv::Mat cv_R = cv::Mat(3, 3, CV_32F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      cv_R.at<float>(i, j) = R(i, j);
    }
  return cv_R.clone();
}

Eigen::Vector3d MatEigenConverter::MatToVector3d(const cv::Mat& t) {
  Eigen::Vector3d eigen_t = Eigen::Vector3d::Zero();
  for (int i = 0; i < 3; i++) {
    eigen_t[i] = t.at<float>(i, 0);
  }
  return eigen_t;
}

cv::Mat MatEigenConverter::Vector3dToMat(const Eigen::Vector3d& t) {
  cv::Mat cv_t = cv::Mat(3, 1, CV_32F);
  for (int i = 0; i < 3; i++) {
    cv_t.at<float>(i, 0) = t[i];
  }
  return cv_t.clone();
}

Eigen::Matrix<double, 7, 1> MatEigenConverter::MatToMatrix_7_1(
    const cv::Mat& pose) {
  Eigen::Matrix<double, 7, 1> Tcw_7_1;
  Eigen::Matrix3d R;
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

cv::Mat MatEigenConverter::Matrix_7_1_ToMat(
    const Eigen::Matrix<double, 7, 1>& Tcw_7_1) {
  Eigen::Quaterniond q(Tcw_7_1[6], Tcw_7_1[3], Tcw_7_1[4], Tcw_7_1[5]);
  Eigen::Matrix3d R = q.normalized().toRotationMatrix();
  Eigen::Matrix4d _pose = Eigen::Matrix4d::Identity();
  _pose.block<3, 3>(0, 0) = R;
  _pose.block<3, 1>(0, 3) = Tcw_7_1.block<3, 1>(0, 0);

  cv::Mat pose = Matrix4dToMat(_pose);
  return pose.clone();
}

cv::Mat MatEigenConverter::Sim3ToMat(const Sim3& sim) {
  Eigen::Matrix3d R = sim.rotation().toRotationMatrix();
  Eigen::Vector3d t = sim.translation();
  double s = sim.scale();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = s * R;
  T.block<3, 1>(0, 3) = t;
  return Matrix4dToMat(T);
}

std::vector<cv::Mat> MatEigenConverter::toDescriptorVector(
    const cv::Mat& Descriptors) {
  std::vector<cv::Mat> vDesc;
  vDesc.reserve(Descriptors.rows);
  for (int j = 0; j < Descriptors.rows; j++)
    vDesc.push_back(Descriptors.row(j));

  return vDesc;
}
