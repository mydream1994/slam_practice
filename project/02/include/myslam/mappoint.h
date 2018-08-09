#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/camera.h"

namespace myslam
{
  class Frame;
  //路标点,也就是Map类的单位成员,许多mappoint构成了一个Map
  class MapPoint
  {
  public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long id_;   //ID
    Vector3d pos_;   //Position in world
    Vector3d norm_;   //Normal of viewing direction
    Mat descriptor_;  //Descriptor for matching
    //一个点被观测到的次数
    int observed_times_;   //being observed by feature matching logo
    //被匹配的次数
    int correct_times_;    //being an inliner in pose estimation
    
    MapPoint();
    MapPoint(long id,Vector3d position,Vector3d norm);
    
    //factory function
    static MapPoint::Ptr createMapPoint();
  };
}

#endif