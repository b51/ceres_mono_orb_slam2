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

#include "KeyFrameDatabase.h"

#include <mutex>

#include "KeyFrame.h"
#include "ORBVocabulary.h"
#include "lib/DBoW2/DBoW2/BowVector.h"


namespace ORB_SLAM2 {

KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary& voc)
    : orb_vocabulary_(&voc) {
  inverted_files_.resize(voc.size());
}

void KeyFrameDatabase::add(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_);

  for (DBoW2::BowVector::const_iterator vit = keyframe->bow_vector_.begin(),
                                        vend = keyframe->bow_vector_.end();
       vit != vend; vit++)
    inverted_files_[vit->first].push_back(keyframe);
}

void KeyFrameDatabase::erase(KeyFrame* keyframe) {
  unique_lock<mutex> lock(mutex_);

  // Erase elements in the Inverse File for the entry
  for (DBoW2::BowVector::const_iterator vit = keyframe->bow_vector_.begin(),
                                        vend = keyframe->bow_vector_.end();
       vit != vend; vit++) {
    // List of keyframes that share the word
    std::list<KeyFrame*>& keyframes = inverted_files_[vit->first];

    for (std::list<KeyFrame*>::iterator lit = keyframes.begin(),
                                        lend = keyframes.end();
         lit != lend; lit++) {
      if (keyframe == *lit) {
        keyframes.erase(lit);
        break;
      }
    }
  }
}

void KeyFrameDatabase::clear() {
  inverted_files_.clear();
  inverted_files_.resize(orb_vocabulary_->size());
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(
    KeyFrame* keyframe, float minScore) {
  std::set<KeyFrame*> connected_keyframes = keyframe->GetConnectedKeyFrames();
  std::list<KeyFrame*> keyframes_sharing_words;

  // Search all keyframes that share a word with current keyframes
  // Discard keyframes connected to the query keyframe
  {
    unique_lock<mutex> lock(mutex_);

    for (DBoW2::BowVector::const_iterator vit = keyframe->bow_vector_.begin(),
                                          vend = keyframe->bow_vector_.end();
         vit != vend; vit++) {
      list<KeyFrame*>& keyframes = inverted_files_[vit->first];

      for (list<KeyFrame*>::iterator lit = keyframes.begin(),
                                     lend = keyframes.end();
           lit != lend; lit++) {
        KeyFrame* ikeyframe = *lit;
        if (ikeyframe->n_loop_query_ != keyframe->id_) {
          ikeyframe->n_loop_words_ = 0;
          if (!connected_keyframes.count(ikeyframe)) {
            ikeyframe->n_loop_query_ = keyframe->id_;
            keyframes_sharing_words.push_back(ikeyframe);
          }
        }
        ikeyframe->n_loop_words_++;
      }
    }
  }

  if (keyframes_sharing_words.empty()) {
    return std::vector<KeyFrame*>();
  }

  std::list<std::pair<float, KeyFrame*> > score_and_matches;

  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (list<KeyFrame*>::iterator lit = keyframes_sharing_words.begin(),
                                 lend = keyframes_sharing_words.end();
       lit != lend; lit++) {
    if ((*lit)->n_loop_words_ > maxCommonWords)
      maxCommonWords = (*lit)->n_loop_words_;
  }

  int minCommonWords = maxCommonWords * 0.8f;

  int nscores = 0;

  // Compute similarity score. Retain the matches whose score is higher than
  // minScore
  for (list<KeyFrame*>::iterator lit = keyframes_sharing_words.begin(),
                                 lend = keyframes_sharing_words.end();
       lit != lend; lit++) {
    KeyFrame* ikeyframe = *lit;

    if (ikeyframe->n_loop_words_ > minCommonWords) {
      nscores++;

      float score =
          orb_vocabulary_->score(keyframe->bow_vector_, ikeyframe->bow_vector_);

      ikeyframe->loop_score_ = score;
      if (score >= minScore)
        score_and_matches.push_back(std::make_pair(score, ikeyframe));
    }
  }

  if (score_and_matches.empty()) {
    return std::vector<KeyFrame*>();
  }

  std::list<std::pair<float, KeyFrame*> > acc_score_and_matches;
  float bestAccScore = minScore;

  // Lets now accumulate score by covisibility
  for (std::list<std::pair<float, KeyFrame*> >::iterator
           it = score_and_matches.begin(),
           itend = score_and_matches.end();
       it != itend; it++) {
    KeyFrame* ikeyframe = it->second;
    std::vector<KeyFrame*> neighs = ikeyframe->GetBestCovisibilityKeyFrames(10);

    float bestScore = it->first;
    float accScore = it->first;
    KeyFrame* best_keyframe = ikeyframe;
    for (std::vector<KeyFrame*>::iterator vit = neighs.begin(),
                                          vend = neighs.end();
         vit != vend; vit++) {
      KeyFrame* jkeyframe = *vit;
      if (jkeyframe->n_loop_query_ == keyframe->id_ &&
          jkeyframe->n_loop_words_ > minCommonWords) {
        accScore += jkeyframe->loop_score_;
        if (jkeyframe->loop_score_ > bestScore) {
          best_keyframe = jkeyframe;
          bestScore = jkeyframe->loop_score_;
        }
      }
    }

    acc_score_and_matches.push_back(std::make_pair(accScore, best_keyframe));
    if (accScore > bestAccScore) {
      bestAccScore = accScore;
    }
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;

  std::set<KeyFrame*> alread_added_keyframes;
  std::vector<KeyFrame*> loop_candidates_;
  loop_candidates_.reserve(acc_score_and_matches.size());

  for (list<pair<float, KeyFrame*> >::iterator
           it = acc_score_and_matches.begin(),
           itend = acc_score_and_matches.end();
       it != itend; it++) {
    if (it->first > minScoreToRetain) {
      KeyFrame* ikeyframe = it->second;
      if (!alread_added_keyframes.count(ikeyframe)) {
        loop_candidates_.push_back(ikeyframe);
        alread_added_keyframes.insert(ikeyframe);
      }
    }
  }

  return loop_candidates_;
}

vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(
    Frame* frame) {
  std::list<KeyFrame*> keyframes_sharing_words;

  // Search all keyframes that share a word with current frame
  {
    unique_lock<mutex> lock(mutex_);

    for (DBoW2::BowVector::const_iterator vit = frame->bow_vector_.begin(),
                                          vend = frame->bow_vector_.end();
         vit != vend; vit++) {
      std::list<KeyFrame*>& keyframes = inverted_files_[vit->first];

      for (std::list<KeyFrame*>::iterator lit = keyframes.begin(),
                                          lend = keyframes.end();
           lit != lend; lit++) {
        KeyFrame* ikeyframe = *lit;
        if (ikeyframe->reloc_query_ != frame->id_) {
          ikeyframe->n_reloc_words_ = 0;
          ikeyframe->reloc_query_ = frame->id_;
          keyframes_sharing_words.push_back(ikeyframe);
        }
        ikeyframe->n_reloc_words_++;
      }
    }
  }
  if (keyframes_sharing_words.empty()) {
    return std::vector<KeyFrame*>();
  }

  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (list<KeyFrame*>::iterator lit = keyframes_sharing_words.begin(),
                                 lend = keyframes_sharing_words.end();
       lit != lend; lit++) {
    if ((*lit)->n_reloc_words_ > maxCommonWords)
      maxCommonWords = (*lit)->n_reloc_words_;
  }

  int minCommonWords = maxCommonWords * 0.8f;

  std::list<std::pair<float, KeyFrame*> > score_and_matches;

  int nscores = 0;

  // Compute similarity score.
  for (list<KeyFrame*>::iterator lit = keyframes_sharing_words.begin(),
                                 lend = keyframes_sharing_words.end();
       lit != lend; lit++) {
    KeyFrame* ikeyframe = *lit;

    if (ikeyframe->n_reloc_words_ > minCommonWords) {
      nscores++;
      float score =
          orb_vocabulary_->score(frame->bow_vector_, ikeyframe->bow_vector_);
      ikeyframe->reloc_score_ = score;
      score_and_matches.push_back(std::make_pair(score, ikeyframe));
    }
  }

  if (score_and_matches.empty()) {
    return std::vector<KeyFrame*>();
  }

  std::list<std::pair<float, KeyFrame*> > acc_score_and_matches;
  float bestAccScore = 0;

  // Lets now accumulate score by covisibility
  for (list<pair<float, KeyFrame*> >::iterator it = score_and_matches.begin(),
                                               itend = score_and_matches.end();
       it != itend; it++) {
    KeyFrame* ikeyframe = it->second;
    std::vector<KeyFrame*> neighs = ikeyframe->GetBestCovisibilityKeyFrames(10);

    float bestScore = it->first;
    float accScore = bestScore;
    KeyFrame* best_keyframe = ikeyframe;
    for (vector<KeyFrame*>::iterator vit = neighs.begin(), vend = neighs.end();
         vit != vend; vit++) {
      KeyFrame* jkeyframe = *vit;
      if (jkeyframe->reloc_query_ != frame->id_) continue;

      accScore += jkeyframe->reloc_score_;
      if (jkeyframe->reloc_score_ > bestScore) {
        best_keyframe = jkeyframe;
        bestScore = jkeyframe->reloc_score_;
      }
    }
    acc_score_and_matches.push_back(std::make_pair(accScore, best_keyframe));
    if (accScore > bestAccScore) {
      bestAccScore = accScore;
    }
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;
  std::set<KeyFrame*> alread_added_keyframes;
  std::vector<KeyFrame*> reloc_candidates;
  reloc_candidates.reserve(acc_score_and_matches.size());
  for (std::list<std::pair<float, KeyFrame*> >::iterator
           it = acc_score_and_matches.begin(),
           itend = acc_score_and_matches.end();
       it != itend; it++) {
    const float& score = it->first;
    if (score > minScoreToRetain) {
      KeyFrame* ikeyframe = it->second;
      if (!alread_added_keyframes.count(ikeyframe)) {
        reloc_candidates.push_back(ikeyframe);
        alread_added_keyframes.insert(ikeyframe);
      }
    }
  }

  return reloc_candidates;
}
}  // namespace ORB_SLAM2
