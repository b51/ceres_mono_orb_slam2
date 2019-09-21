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
