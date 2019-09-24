/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: CeresSim3.cc
 *
 *          Created On: Tue 24 Sep 2019 05:58:09 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#include "CeresSim3.h"

Sim3 operator*(const Sim3& lhs, const Sim3& rhs) {
  return Sim3(
      lhs.scale() * rhs.scale(), lhs.rotation() * rhs.rotation(),
      lhs.scale() * (lhs.rotation() * rhs.translation()) + lhs.translation());
}

std::ostream& operator<<(std::ostream& out_str, const Sim3& sim3) {
  out_str << " [ " << std::endl;
  out_str << "scale: " << sim3.scale() << std::endl;
  out_str << "q.x, q.y, q.z, q.w: " << sim3.rotation().coeffs() << std::endl;
  out_str << "t.x, t.y, t.z: " << sim3.translation() << std::endl;
  out_str << " ] " << std::endl;
  return out_str;
}
