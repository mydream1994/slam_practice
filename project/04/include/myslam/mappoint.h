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
    static unsigned long factory_id_;   //factory id
    bool good_;      //wheter a good point
    Vector3d pos_;   //Position in world
    Vector3d norm_;   //Normal of viewing direction
    Mat descriptor_;  //Descriptor for matching
    
    list<Frame*> observed_frames_;   //key-frame that can observe this point
    /*
    //一个点被观测到的次数
    int observed_times_;   //being observed by feature matching logo
    //被匹配的次数
    int correct_times_;    //being an inliner in pose estimation
    */
    
    int matched_times_;    //being an inliner in pose estimation(路标点被当前帧匹配的次数)
    int visible_times_;   //being visible in current frame(路标点在视野的范围内的次数)
    
    MapPoint();
    MapPoint(long id,Vector3d position,Vector3d norm,Frame* frame=nullptr,const Mat& descriptor=Mat());
    
    inline cv::Point3f getPositionCV() const{
      return cv::Point3f(pos_(0,0),pos_(1,0),pos_(2,0));
    }
    //factory function
    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint(
      const Vector3d& pos_world,
      const Vector3d& norm,
      const Mat&  descriptor,
      Frame* frame);
  };
}

#endif