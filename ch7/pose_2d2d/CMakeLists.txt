cmake_minimum_required( VERSION 2.8)
project( pose_2d2d)

set( CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package( OpenCV 3.2 REQUIRED)
#find_package( OpenCV REQUIRED)     #use this if in OpenCV2

include_directories( ${OpenCV_INCLUDE_DIRS} "/usr/include/eigen3")

add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS})