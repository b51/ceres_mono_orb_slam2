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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>

namespace ORB_SLAM2 {
FrameDrawer::FrameDrawer(Map* map) : map_(map) {
  state_ = Tracking::SYSTEM_NOT_READY;
  image_ = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat FrameDrawer::DrawFrame() {
  cv::Mat img;
  // Initialization: KeyPoints in reference frame
  std::vector<cv::KeyPoint> init_keypoints;
  // Initialization: correspondeces with reference keypoints
  std::vector<int> matches;
  // KeyPoints in current frame
  std::vector<cv::KeyPoint> current_keypoints;
  // Tracked MapPoints in current frame
  std::vector<bool> do_vo, do_map;
  // Tracking state
  int state;

  // Copy variables within scoped mutex
  {
    unique_lock<mutex> lock(mutex_);
    state = state_;
    if (state_ == Tracking::SYSTEM_NOT_READY) state_ = Tracking::NO_IMAGES_YET;

    image_.copyTo(img);

    if (state_ == Tracking::NOT_INITIALIZED) {
      current_keypoints = current_keypoints_;
      init_keypoints = init_keypoints_;
      matches = init_matches_;
    } else if (state_ == Tracking::OK) {
      current_keypoints = current_keypoints_;
      do_vo = do_vo_;
      do_map = do_map_;
    } else if (state_ == Tracking::LOST) {
      current_keypoints = current_keypoints_;
    }
  }  // destroy scoped mutex -> release mutex

  if (img.channels() < 3)  // this should be always true
    cv::cvtColor(img, img, CV_GRAY2BGR);

  // Draw
  if (state == Tracking::NOT_INITIALIZED) {
    // INITIALIZING
    for (unsigned int i = 0; i < matches.size(); i++) {
      if (matches[i] >= 0) {
        cv::line(img, init_keypoints[i].pt, current_keypoints[matches[i]].pt,
                 cv::Scalar(0, 255, 0));
      }
    }
  } else if (state == Tracking::OK) {
    // TRACKING
    ntracked_ = 0;
    ntracked_vo_ = 0;
    const float r = 5;
    const int n = current_keypoints.size();
    for (int i = 0; i < n; i++) {
      if (do_vo[i] || do_map[i]) {
        cv::Point2f pt1, pt2;
        pt1.x = current_keypoints[i].pt.x - r;
        pt1.y = current_keypoints[i].pt.y - r;
        pt2.x = current_keypoints[i].pt.x + r;
        pt2.y = current_keypoints[i].pt.y + r;

        // This is a match to a MapPoint in the map
        if (do_map[i]) {
          cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
          cv::circle(img, current_keypoints[i].pt, 2, cv::Scalar(0, 255, 0),
                     -1);
          ntracked_++;
        } else  // This is match to a "visual odometry" MapPoint created in the
                // last frame
        {
          cv::rectangle(img, pt1, pt2, cv::Scalar(255, 0, 0));
          cv::circle(img, current_keypoints[i].pt, 2, cv::Scalar(255, 0, 0),
                     -1);
          ntracked_vo_++;
        }
      }
    }
  }

  cv::Mat imgWithInfo;
  DrawTextInfo(img, state, imgWithInfo);

  return imgWithInfo;
}

void FrameDrawer::DrawTextInfo(cv::Mat& img, int state, cv::Mat& imgText) {
  stringstream s;
  if (state == Tracking::NO_IMAGES_YET)
    s << " WAITING FOR IMAGES";
  else if (state == Tracking::NOT_INITIALIZED)
    s << " TRYING TO INITIALIZE ";
  else if (state == Tracking::OK) {
    if (!do_only_tracking_)
      s << "SLAM MODE |  ";
    else
      s << "LOCALIZATION | ";
    int nKFs = map_->KeyFramesInMap();
    int nMPs = map_->MapPointsInMap();
    s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << ntracked_;
    if (ntracked_vo_ > 0) {
      s << ", + VO matches: " << ntracked_vo_;
    }
  } else if (state == Tracking::LOST) {
    s << " TRACK LOST. TRYING TO RELOCALIZE ";
  } else if (state == Tracking::SYSTEM_NOT_READY) {
    s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
  }

  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

  imgText = cv::Mat(img.rows + textSize.height + 10, img.cols, img.type());
  img.copyTo(imgText.rowRange(0, img.rows).colRange(0, img.cols));
  imgText.rowRange(img.rows, imgText.rows) =
      cv::Mat::zeros(textSize.height + 10, img.cols, img.type());
  cv::putText(imgText, s.str(), cv::Point(5, imgText.rows - 5),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

void FrameDrawer::Update(Tracking* tracker) {
  unique_lock<mutex> lock(mutex_);
  tracker->img_gray_.copyTo(image_);
  current_keypoints_ = tracker->current_frame_.keypoints_;
  N_ = current_keypoints_.size();
  do_vo_ = std::vector<bool>(N_, false);
  do_map_ = std::vector<bool>(N_, false);
  do_only_tracking_ = tracker->do_only_tracking_;

  if (tracker->last_processed_state_ == Tracking::NOT_INITIALIZED) {
    init_keypoints_ = tracker->init_frame_.keypoints_;
    init_matches_ = tracker->init_matches_;
  } else if (tracker->last_processed_state_ == Tracking::OK) {
    for (int i = 0; i < N_; i++) {
      MapPoint* map_point = tracker->current_frame_.map_points_[i];
      if (map_point) {
        if (!tracker->current_frame_.is_outliers_[i]) {
          if (map_point->Observations() > 0)
            do_map_[i] = true;
          else
            do_vo_[i] = true;
        }
      }
    }
  }
  state_ = static_cast<int>(tracker->last_processed_state_);
}
}  // namespace ORB_SLAM2
