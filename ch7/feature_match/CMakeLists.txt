cmake_minimum_required( VERSION 2.8)
project( feature_match)

set( CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package( OpenCV 3.2 REQUIRED)
#find_package( OpenCV REQUIRED)     #use this if in OpenCV2

include_directories( ${OpenCV_INCLUDE_DIRS} "/usr/include/eigen3")

add_executable( feature_extraction feature_extraction.cpp)
target_link_libraries( feature_extraction ${OpenCV_LIBS})