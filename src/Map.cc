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

#include "Map.h"

#include <mutex>

namespace ORB_SLAM2 {

Map::Map() : max_keyframe_id_(0), big_change_index_(0) {}

void Map::AddKeyFrame(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_map_);
  keyframes_.insert(keyframe);
  if (keyframe->id_ > max_keyframe_id_) {
    max_keyframe_id_ = keyframe->id_;
  }
}

void Map::AddMapPoint(MapPoint* map_point) {
  unique_lock<mutex> lock(mutex_map_);
  map_points_.insert(map_point);
}

void Map::EraseMapPoint(MapPoint* map_point) {
  unique_lock<mutex> lock(mutex_map_);
  map_points_.erase(map_point);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_map_);
  keyframes_.erase(keyframe);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint*>& map_points) {
  unique_lock<mutex> lock(mutex_map_);
  reference_map_points_ = map_points;
}

void Map::InformNewBigChange() {
  unique_lock<mutex> lock(mutex_map_);
  big_change_index_++;
}

int Map::GetLastBigChangeIdx() {
  unique_lock<mutex> lock(mutex_map_);
  return big_change_index_;
}

vector<KeyFrame*> Map::GetAllKeyFrames() {
  unique_lock<mutex> lock(mutex_map_);
  return std::vector<KeyFrame*>(keyframes_.begin(), keyframes_.end());
}

vector<MapPoint*> Map::GetAllMapPoints() {
  unique_lock<mutex> lock(mutex_map_);
  return std::vector<MapPoint*>(map_points_.begin(), map_points_.end());
}

long unsigned int Map::MapPointsInMap() {
  unique_lock<mutex> lock(mutex_map_);
  return map_points_.size();
}

long unsigned int Map::KeyFramesInMap() {
  unique_lock<mutex> lock(mutex_map_);
  return keyframes_.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints() {
  unique_lock<mutex> lock(mutex_map_);
  return reference_map_points_;
}

long unsigned int Map::GetMaxKFid() {
  unique_lock<mutex> lock(mutex_map_);
  return max_keyframe_id_;
}

void Map::clear() {
  for (std::set<MapPoint*>::iterator sit = map_points_.begin(),
                                     send = map_points_.end();
       sit != send; sit++)
    delete *sit;

  for (std::set<KeyFrame*>::iterator sit = keyframes_.begin(),
                                     send = keyframes_.end();
       sit != send; sit++)
    delete *sit;

  map_points_.clear();
  keyframes_.clear();
  max_keyframe_id_ = 0;
  reference_map_points_.clear();
  keyframe_origins_.clear();
}
}  // namespace ORB_SLAM2
