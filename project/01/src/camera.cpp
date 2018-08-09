#include "myslam/camera.h"

namespace myslam {
Camera::Camera()
{

}

//世界坐标到相机坐标
Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d& p_w,const SE3& T_c_w)
{
  return T_c_w*p_w;
}

//相机坐标到世界坐标
Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d& p_c,const SE3& T_c_w)
{
  return T_c_w.inverse()*p_c; 
}

Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d& p_c)
{
  return Vector2d(
    fx_*p_c(0,0)/p_c(2,0)+cx_,
    fy_*p_c(1,0)/p_c(2,0)+cy_
  );
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d& p_p,double depth)
{
  return Vector3d(
    (p_p(0,0)-cx_)*depth/fx_,
    (p_p(1,0)-cy_)*depth/fy_,
    depth
  );
}

Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d& p_p,const SE3& T_c_w,double depth)
{
  return camera2world(pixel2camera(p_p,depth),T_c_w);
}
  
Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d& p_w,const SE3& T_c_w)
{
  return camera2pixel(world2camera(p_w,T_c_w));
}
}