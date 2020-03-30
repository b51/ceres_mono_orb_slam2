/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: main.cc
 *
 *          Created On: Tue 03 Sep 2019 06:19:56 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>

#include "MonoORBSlam.h"

DEFINE_string(voc, "vocabulary/ORBvoc.txt", "Path to ORB vocabulary");
DEFINE_string(config, "configs/TUM2.yaml", "Path to camera intrinsics");
DEFINE_string(images, "rgbd_dataset_freiburg2_desk", "Path to images folder");

void LoadImages(const std::string& strFile, std::vector<std::string>& vstrImageFilenames,
                std::vector<double>& vTimestamps) {
  std::ifstream f(strFile.c_str());
  if (!f.good()) LOG(FATAL) << strFile << " not exists";

  // skip first three lines
  std::string s0;
  getline(f, s0);
  getline(f, s0);
  getline(f, s0);

  while (!f.eof()) {
    std::string s;
    getline(f, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      double t;
      std::string sRGB;
      ss >> t;
      vTimestamps.push_back(t);
      ss >> sRGB;
      vstrImageFilenames.push_back(sRGB);
    }
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;

  LOG(WARNING) << "Usage: " << argv[0]
               << " --voc ORBvoc.txt --config config.yaml --images imgs_dir";
  std::vector<std::string> string_image_filenames;
  std::vector<double> timestamps;
  std::string string_file = std::string(FLAGS_images) + "/rgb.txt";
  LoadImages(string_file, string_image_filenames, timestamps);

  int n_images = string_image_filenames.size();

  ORB_SLAM2::MonoORBSlam slam(FLAGS_voc, FLAGS_config, true);

  std::vector<float> times_track;
  times_track.resize(n_images);

  LOG(INFO) << "----------";
  LOG(INFO) << "Start processing sequence ...";
  LOG(INFO) << "Images in the sequence: " << n_images;

  cv::Mat img;
  for (int i = 0; i < n_images; i++) {
    img =
        cv::imread(std::string(FLAGS_images) + "/" + string_image_filenames[i],
                   CV_LOAD_IMAGE_UNCHANGED);
    double tframe = timestamps[i];

    if (img.empty()) {
      LOG(FATAL) << "Failed to load image at: " << std::string(FLAGS_images)
                 << "/" << string_image_filenames[i];
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
    // if (i == n_images - 1) cv::waitKey(0);
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
  slam.SaveMap("map.yaml");

  cv::waitKey(0);
  return 0;
}
