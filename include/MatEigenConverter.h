/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: MatEigenConverter.h
 *
 *          Created On: Fri 20 Sep 2019 06:40:20 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#ifndef MAT_EIGEN_CONVERTER_H_
#define MAT_EIGEN_CONVERTER_H_

#include <Eigen/Geometry>
#include <iostream>
#include <opencv/cv.hpp>

#include "LoopClosing.h"

class MatEigenConverter {
 public:
  static std::vector<cv::Mat> toDescriptorVector(const cv::Mat& Descriptors);

  // Transpose
  static Eigen::Matrix4d MatToMatrix4d(const cv::Mat& T);
  static cv::Mat Matrix4dToMat(const Eigen::Matrix4d& T);

  // Rotation
  static Eigen::Matrix3d MatToMatrix3d(const cv::Mat& R);
  static cv::Mat Matrix3dToMat(const Eigen::Matrix3d& R);

  // translation
  static Eigen::Vector3d MatToVector3d(const cv::Mat& t);
  static cv::Mat Vector3dToMat(const Eigen::Vector3d& t);

  // Matrix_7_1
  static Eigen::Matrix<double, 7, 1> Matrix4dToMatrix_7_1(
      const Eigen::Matrix4d& pose);
  static Eigen::Matrix4d Matrix_7_1_ToMatrix4d(
      const Eigen::Matrix<double, 7, 1>& Tcw_7_1);
};

#endif
