#ifndef CAMERA_H
#define CAMERA_H
#include "myslam/common_include.h"

namespace myslam
{
  //RGB-D camera model
  class Camera
  {
  public:
    //分配一个智能指针
    typedef std::shared_ptr<Camera> Ptr;
    float fx_,fy_,cx_,cy_,depth_scale_;     //Camera intrinsics(内参)
    
    Camera();
    Camera(float fx,float fy,float cx,float cy,float depth_scale=0):
      fx_(fx),fy_(fy),cx_(cx),cy_(cy),depth_scale_(depth_scale){}
      
    //coordinate transform: world,camera,pixel
    Eigen::Vector3d world2camera(const Eigen::Vector3d& p_w,const SE3& T_c_w);
    Eigen::Vector3d camera2world(const Eigen::Vector3d& p_c,const SE3& T_c_w);
    Eigen::Vector2d camera2pixel(const Eigen::Vector3d& p_c);
    //depth=1,默认将像素坐标转化到归一化坐标中
    Eigen::Vector3d pixel2camera(const Eigen::Vector2d& p_p,double depth=1);
    Eigen::Vector3d pixel2world(const Eigen::Vector2d& p_p,const SE3& T_c_w,double depth=1);
    Eigen::Vector2d world2pixel(const Eigen::Vector3d& p_w,const SE3& T_c_w);
  };
}

#endif