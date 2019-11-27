cmake_minimum_required(VERSION 2.8.3)
project(ceres_mono_orb_slam2)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

#########################################################################
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC -march=native")
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fPIC")
endif(CMAKE_BUILD_TYPE STREQUAL "Release")
#########################################################################

#find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
#find_package(OpenMP REQUIRED)

#catkin_package(
#  INCLUDE_DIRS include
#  #CATKIN_DEPENDS
#)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${PROJECT_SOURCE_DIR}
  #${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/MonoORBSlam.cc
  src/Tracking.cc
  src/LocalMapping.cc
  src/LoopClosing.cc
  src/ORBextractor.cc
  src/ORBmatcher.cc
  src/FrameDrawer.cc
  src/MapPoint.cc
  src/KeyFrame.cc
  src/Map.cc
  src/MapDrawer.cc
  src/CeresOptimizer.cc
  src/PnPsolver.cc
  src/Frame.cc
  src/KeyFrameDatabase.cc
  src/Sim3Solver.cc
  src/Initializer.cc
  src/Viewer.cc
  src/MatEigenConverter.cc
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  #OpenMP::OpenMP_CXX
  #${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/lib/DBoW2/lib/libDBoW2.so
  glog
  gflags
)

add_executable(mono_slam
  src/main.cc
)

target_link_libraries(mono_slam
  ${PROJECT_NAME}  
)
