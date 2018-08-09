#ifndef MAP_H
#define MAP_H

#include "myslam/camera.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam
{
  //Map类管理着所有的路标点,并负责添加新路标点,删除不好的路标等工作
  class Map
  {
  public:
    typedef shared_ptr<Map> Ptr;
    //Map类中实际存储了各个关键帧和路标点,即需要随机访问,又需要即时插入和删除,因此
     //使用散列表(hash)来进行存储
    unordered_map<unsigned long,MapPoint::Ptr> map_points_;  //all landmraks  
    unordered_map<unsigned long,Frame::Ptr>    keyframes_;  //all key-frame
    
    Map(){}
    
    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr map_point);
  };
}

#endif