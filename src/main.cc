/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: main.cc
 *
 *          Created On: Tue 03 Sep 2019 06:19:56 PM CST
 *     Licensed under The MIT License [see LICENSE for details]
 *
 ************************************************************************/

#include <glog/logging.h>
#include <iostream>

#include "MonoORBSlam.h"

void LoadImages(const string& strFile, vector<string>& vstrImageFilenames,
                vector<double>& vTimestamps) {
  ifstream f;
  f.open(strFile.c_str());

  // skip first three lines
  string s0;
  getline(f, s0);
  getline(f, s0);
  getline(f, s0);

  while (!f.eof()) {
    string s;
    getline(f, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      string sRGB;
      ss >> t;
      vTimestamps.push_back(t);
      ss >> sRGB;
      vstrImageFilenames.push_back(sRGB);
    }
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  FLAGS_minloglevel = google::INFO;
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;

  std::vector<std::string> string_image_filenames;
  std::vector<double> timestamps;
  std::string string_file = std::string(argv[3]) + "/rgb.txt";
  LoadImages(string_file, string_image_filenames, timestamps);

  int n_images = string_image_filenames.size();

  ORB_SLAM2::MonoORBSlam slam(argv[1], argv[2], true);

  vector<float> times_track;
  times_track.resize(n_images);

  LOG(INFO) << "----------";
  LOG(INFO) << "Start processing sequence ...";
  LOG(INFO) << "Images in the sequence: " << n_images;

  cv::Mat img;
  for (int i = 0; i < n_images; i++) {
    img = cv::imread(std::string(argv[3]) + "/" + string_image_filenames[i],
                     CV_LOAD_IMAGE_UNCHANGED);
    double tframe = timestamps[i];

    if (img.empty()) {
      LOG(FATAL) << "Failed to load image at: " << std::string(argv[3]) << "/"
                 << string_image_filenames[i];
    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    slam.TrackMonocular(img, tframe);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();

    times_track[i] = ttrack;

    double T = 0;
    if (i < n_images - 1) {
      T = timestamps[i + 1] - tframe;
    } else if (i > 0) {
      T = tframe - timestamps[i - 1];
    }

    if (ttrack < T) {
      usleep((T - ttrack) * 1e6);
    }
  }

  slam.Shutdown();
  sort(times_track.begin(), times_track.end());
  float total_time = 0;
  for (int i = 0; i < n_images; i++) {
    total_time += times_track[i];
  }
  LOG(INFO) << "-------------";
  LOG(INFO) << "median tracking time: " << times_track[n_images / 2];
  LOG(INFO) << "mean tracking time: " << total_time / n_images;

  slam.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  return 0;
}
