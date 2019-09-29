/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: Viewer.h
 *
 *          Created On: Wed 04 Sep 2019 06:57:52 PM CST
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

#ifndef VIEWER_H_
#define VIEWER_H_

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "MonoORBSlam.h"
#include "Tracking.h"

#include <mutex>

namespace ORB_SLAM2 {

class Tracking;
class FrameDrawer;
class MapDrawer;
class MonoORBSlam;

class Viewer {
 public:
  Viewer(MonoORBSlam* mono_orb_slam, FrameDrawer* frame_drawer, MapDrawer* map_drawer,
         Tracking* tracker, const string& string_setting_file);

  // Main thread function. Draw points, keyframes, the current camera pose and
  // the last processed frame. Drawing is refreshed according to the camera fps.
  // We use Pangolin.
  void Run();

  void RequestFinish();

  void RequestStop();

  bool isFinished();

  bool isStopped();

  void SetFollowCamera();

  void Release();

 private:
  bool Stop();

  MonoORBSlam* mono_orb_slam_;
  FrameDrawer* frame_drawer_;
  MapDrawer* map_drawer_;
  Tracking* tracker_;

  // 1/fps in ms
  double T_;
  float img_width_, img_height_;

  float view_point_x_, view_point_y_, view_point_z_, view_point_f_;

  bool CheckFinish();
  void SetFinish();
  bool is_finish_requested_;
  bool is_finished_;
  bool is_follow_;
  std::mutex mutex_finish_;

  bool is_stopped_;
  bool is_stop_requested_;
  std::mutex mutex_stop_;
};

}  // namespace ORB_SLAM2
#endif
