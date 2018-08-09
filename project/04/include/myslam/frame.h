#ifndef FRAME_H
#define FRAME_H
#include "myslam/camera.h"

namespace myslam
{
  class MapPoint;
  class Frame
  {
  public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long id_;    //id of this frame
    double time_stamp_;    //when it is recorded(时间戳)
    SE3 T_c_w_;      //transform from world to camera
    Camera::Ptr camera_;   //pinhole RGB-D Camera model
    Mat color_,depth_;
    
  public:
    Frame();
    Frame(long id,double time_stamp=0,SE3 T_c_w=SE3(),Camera::Ptr camera=nullptr,
         Mat color=Mat(),Mat depth=Mat() );
    ~Frame();
    
    //factory function(创建Frame)
    static Frame::Ptr createFrame();
    //find the depth in depth map
    double findDepth(const cv::KeyPoint& kp);
    //Get Camera Center
    //该函数中不可修改类的成员元素
    Eigen::Vector3d getCamCenter() const;
    
    //check if a point in this frame
    bool isInFrame(const Eigen::Vector3d& pt_world);
  };
}
#endif