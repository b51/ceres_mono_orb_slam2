/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Viewer.cc
 *
 *          Created On: Wed 04 Sep 2019 06:59:03 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/
/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <unistd.h>
#include <mutex>

namespace ORB_SLAM2 {

Viewer::Viewer(MonoORBSlam* mono_orb_slam, FrameDrawer* frame_drawer,
               MapDrawer* map_drawer, Tracking* tracker,
               const string& string_setting_file)
    : mono_orb_slam_(mono_orb_slam),
      frame_drawer_(frame_drawer),
      map_drawer_(map_drawer),
      tracker_(tracker),
      is_finish_requested_(false),
      is_finished_(true),
      is_follow_(false),
      is_stopped_(true),
      is_stop_requested_(false) {
  cv::FileStorage fSettings(string_setting_file, cv::FileStorage::READ);
  float fps = fSettings["Camera.fps"];

  if (fps < 1) fps = 30;
  T_ = 1e3 / fps;

  img_width_ = fSettings["Camera.width"];
  img_height_ = fSettings["Camera.height"];
  if (img_width_ < 1 || img_height_ < 1) {
    img_width_ = 640;
    img_height_ = 480;
  }

  view_point_x_ = fSettings["Viewer.ViewpointX"];
  view_point_y_ = fSettings["Viewer.ViewpointY"];
  view_point_z_ = fSettings["Viewer.ViewpointZ"];
  view_point_f_ = fSettings["Viewer.ViewpointF"];
}

void Viewer::Run() {
  is_finished_ = false;
  is_stopped_ = false;

  pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  // Issue specific OpenGl we might need
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", false, true);
  pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
  pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
  pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
  pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false,
                                           true);
  pangolin::Var<bool> menuReset("menu.Reset", false, false);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, view_point_f_, view_point_f_, 512,
                                 389, 0.1, 1000),
      pangolin::ModelViewLookAt(view_point_x_, view_point_y_, view_point_z_, 0,
                                0, 0, 0.0, -1.0, 0.0));

  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175),
                                         1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  cv::namedWindow("ORB-SLAM2: Current Frame");

  bool is_localization_mode = false;

  while (true) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    map_drawer_->GetCurrentOpenGLCameraMatrix(Twc);

    if (menuFollowCamera && is_follow_) {
      s_cam.Follow(Twc);
    } else if (menuFollowCamera && !is_follow_) {
      s_cam.SetModelViewMatrix(
          pangolin::ModelViewLookAt(view_point_x_, view_point_y_, view_point_z_,
                                    0, 0, 0, 0.0, -1.0, 0.0));
      s_cam.Follow(Twc);
      is_follow_ = true;
    } else if (!menuFollowCamera && is_follow_) {
      is_follow_ = false;
    }

    if (menuLocalizationMode && !is_localization_mode) {
      mono_orb_slam_->ActivateLocalizationMode();
      is_localization_mode = true;
    } else if (!menuLocalizationMode && is_localization_mode) {
      mono_orb_slam_->DeactivateLocalizationMode();
      is_localization_mode = false;
    }

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    map_drawer_->DrawCurrentCamera(Twc);
    if (menuShowKeyFrames || menuShowGraph)
      map_drawer_->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
    if (menuShowPoints) map_drawer_->DrawMapPoints();

    pangolin::FinishFrame();

    cv::Mat img = frame_drawer_->DrawFrame();
    cv::imshow("ORB-SLAM2: Current Frame", img);
    cv::waitKey(T_);

    if (menuReset) {
      menuShowGraph = true;
      menuShowKeyFrames = true;
      menuShowPoints = true;
      menuLocalizationMode = false;
      if (is_localization_mode) {
        mono_orb_slam_->DeactivateLocalizationMode();
      }
      is_localization_mode = false;
      is_follow_ = true;
      menuFollowCamera = true;
      mono_orb_slam_->Reset();
      menuReset = false;
    }

    if (Stop()) {
      while (isStopped()) {
        usleep(3000);
      }
    }

    if (CheckFinish()) {
      break;
    }
  }

  SetFinish();
}

void Viewer::RequestFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  is_finish_requested_ = true;
}

bool Viewer::CheckFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  return is_finish_requested_;
}

void Viewer::SetFinish() {
  unique_lock<mutex> lock(mutex_finish_);
  is_finished_ = true;
}

bool Viewer::isFinished() {
  unique_lock<mutex> lock(mutex_finish_);
  return is_finished_;
}

void Viewer::RequestStop() {
  unique_lock<mutex> lock(mutex_stop_);
  if (!is_stopped_) is_stop_requested_ = true;
}

bool Viewer::isStopped() {
  unique_lock<mutex> lock(mutex_stop_);
  return is_stopped_;
}

bool Viewer::Stop() {
  unique_lock<mutex> lock(mutex_stop_);
  unique_lock<mutex> lock2(mutex_finish_);

  if (is_finish_requested_)
    return false;
  else if (is_stop_requested_) {
    is_stopped_ = true;
    is_stop_requested_ = false;
    return true;
  }

  return false;
}

void Viewer::SetFollowCamera() {
  unique_lock<mutex> lock(mutex_stop_);
  is_follow_ = true;
}

void Viewer::Release() {
  unique_lock<mutex> lock(mutex_stop_);
  is_stopped_ = false;
}
}  // namespace ORB_SLAM2
