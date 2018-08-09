#include "myslam/g2o_types.h"

namespace myslam {
  
void EdgeProjectXYZRGBD::computeError()
{
     //待优化的变量
     const g2o::VertexSBAPointXYZ* point = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
     const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
     //_measurement is p,point is p'
     //pose->estimate().map(_point) 中用estimate()估计一个值后,然后用映射函数就是旋转加平移，将
     //其_point映射到一个相机坐标系
     _error = _measurement - pose->estimate().map(point->estimate());
}

void EdgeProjectXYZRGBD::linearizeOplus()
{
  g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
  g2o::SE3Quat T(pose->estimate());  //得到变换矩阵
  g2o::VertexSBAPointXYZ* point = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
  Eigen::Vector3d xyz = point->estimate();
  Eigen::Vector3d xyz_trans = T.map(xyz);
  //p'(x,y,x)为估计值
  double x=xyz_trans[0];
  double y=xyz_trans[1];
  double z=xyz_trans[2];
  
  //观测相机方程关于特征点的导数矩阵
  _jacobianOplusXi = -T.rotation().toRotationMatrix(); //四元数提出旋转向量,再转换为矩阵
  
  //3x6的雅克比矩阵 误差对应的导数  优化变量更新增量
    _jacobianOplusXj(0,0)=0;
    _jacobianOplusXj(0,1)=-z;
    _jacobianOplusXj(0,2)=y;
    _jacobianOplusXj(0,3)=-1;
    _jacobianOplusXj(0,4)=0;
    _jacobianOplusXj(0,5)=0;
    
    _jacobianOplusXj(1,0)=z;
    _jacobianOplusXj(1,1)=0;
    _jacobianOplusXj(1,2)=-x;
    _jacobianOplusXj(1,3)=0;
    _jacobianOplusXj(1,4)=-1;
    _jacobianOplusXj(1,5)=0;
    
    _jacobianOplusXj(2,0)=-y;
    _jacobianOplusXj(2,1)=x;
    _jacobianOplusXj(2,2)=0;
    _jacobianOplusXj(2,3)=0;
    _jacobianOplusXj(2,4)=0;
    _jacobianOplusXj(2,5)=-1;
   
}

void EdgeProjectXYZRGBDPoseOnly::computeError()
{
   const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
   _error = _measurement - pose->estimate().map(point_);
}

void EdgeProjectXYZRGBDPoseOnly::linearizeOplus()
{
   g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
   g2o::SE3Quat T(pose->estimate());
   Vector3d xyz_trans = T.map(point_);
   //估计值
   double x=xyz_trans[0];
   double y=xyz_trans[1];
   double z=xyz_trans[2];
   
        //3x6的雅克比矩阵 误差对应的导数  优化变量更新增量
    _jacobianOplusXi(0,0)=0;
    _jacobianOplusXi(0,1)=-z;
    _jacobianOplusXi(0,2)=y;
    _jacobianOplusXi(0,3)=-1;
    _jacobianOplusXi(0,4)=0;
    _jacobianOplusXi(0,5)=0;
    
    _jacobianOplusXi(1,0)=z;
    _jacobianOplusXi(1,1)=0;
    _jacobianOplusXi(1,2)=-x;
    _jacobianOplusXi(1,3)=0;
    _jacobianOplusXi(1,4)=-1;
    _jacobianOplusXi(1,5)=0;
    
    _jacobianOplusXi(2,0)=-y;
    _jacobianOplusXi(2,1)=x;
    _jacobianOplusXi(2,2)=0;
    _jacobianOplusXi(2,3)=0;
    _jacobianOplusXi(2,4)=0;
    _jacobianOplusXi(2,5)=-1;
}

void EdgeProjectXYZ2UVPoseOnly::computeError()
{
   const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    //计算重投影误差
    _error = _measurement - camera_->camera2pixel(pose->estimate().map(point_));
}

void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
{
  g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
  g2o::SE3Quat T(pose->estimate());
  Vector3d xyz_trans = T.map(point_);
  double x=xyz_trans[0];
  double y=xyz_trans[1];
  double z=xyz_trans[2];
  double z_2 = z*z;
  
  _jacobianOplusXi(0,0) = x*y/z_2*camera_->fx_;
  _jacobianOplusXi(0,1) = -(1+(x*x/z_2))*camera_->fx_;
  _jacobianOplusXi(0,2) = y/z * camera_->fx_;
  _jacobianOplusXi(0,3) = -1./z*camera_->fx_;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x/z_2*camera_->fx_;
  
  _jacobianOplusXi(1,0) = (1+y*y/z_2)*camera_->fy_;
  _jacobianOplusXi(1,1) = -x*y/z_2*camera_->fy_;
  _jacobianOplusXi(1,2) = -x/z*camera_->fy_;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -1./z*camera_->fy_;
  _jacobianOplusXi(1,5) = y/z_2*camera_->fy_;
    
}

}