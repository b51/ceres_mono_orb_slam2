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

#ifndef KEY_FRAME_DATA_BASE_H_
#define KEY_FRAME_DATA_BASE_H_

#include <list>
#include <mutex>
#include <set>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "ORBVocabulary.h"

namespace ORB_SLAM2 {
class KeyFrame;
class Frame;

class KeyFrameDatabase {
 public:
  KeyFrameDatabase(const ORBVocabulary& voc);

  void add(KeyFrame* keyframe);

  void erase(KeyFrame* keyframe);

  void clear();

  // Loop Detection
  std::vector<KeyFrame*> DetectLoopCandidates(KeyFrame* keyframe,
                                              float minScore);

  // Relocalization
  std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* frame);

 protected:
  // Associated vocabulary
  const ORBVocabulary* orb_vocabulary_;

  // Inverted file
  std::vector<list<KeyFrame*> > inverted_files_;

  // Mutex
  std::mutex mutex_;
};
}  // namespace ORB_SLAM2

#endif
