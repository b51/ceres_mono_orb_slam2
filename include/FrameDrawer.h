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

#ifndef FRAMEDRAWER_H_
#define FRAMEDRAWER_H_

#include "Map.h"
#include "MapPoint.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <mutex>

namespace ORB_SLAM2 {

class Tracking;
class Viewer;

class FrameDrawer {
 public:
  FrameDrawer(Map* map);

  // Update info from the last processed frame.
  void Update(Tracking* tracker);

  // Draw last processed frame.
  cv::Mat DrawFrame();

 protected:
  void DrawTextInfo(cv::Mat& img, int state, cv::Mat& imText);

  // Info of the frame to be drawn
  cv::Mat image_;
  int N_;
  vector<cv::KeyPoint> current_keypoints_;
  vector<bool> do_map_, do_vo_;
  bool do_only_tracking_;
  int ntracked_, ntracked_vo_;
  vector<cv::KeyPoint> init_keypoints_;
  vector<int> init_matches_;
  int state_;

  Map* map_;

  std::mutex mutex_;
};
}  // namespace ORB_SLAM2

#endif
