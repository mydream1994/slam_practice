#include <Eigen/Core>
#include <boost/concept_check.hpp>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"

//定义相机位姿顶点类,由于相机内参也作为优化变量,所以包含了:
//焦距f,畸变系数k1,k2,3个参数的平移,3个参数的旋转。一共九个量,9维,类型为Eigen::VectorXd
class VertexCameraBAL:public g2o::BaseVertex<9,Eigen::VectorXd>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  VertexCameraBAL(){}
  
  //这里的读写功能函数就需要用了，参数分别是输入输出流类型实例的引用
  virtual bool read(std::istream& )
  {
    return false;
  }
  
  virtual bool write(std::ostream&) const
  {
    return false;
  }
  //设定顶点的初始值
  virtual void setToOriginImpl(){}
  //增量函数,增量为传进的参数update,这里是9个double值,所以就是double类型的指针(就是数组)
  virtual void oplusImpl(const double* update)
  {
    //将增量数组构造成增量矩阵
    //由于update是个double类型数组,而增量需要的是个矩阵
    //所以用update构造一个增量矩阵v,下面更新估计值,直接将v加上就好了
    Eigen::VectorXd::ConstMapType v(update,VertexCameraBAL::Dimension);
    //将增量矩阵加到估计上
    _estimate += v;
  }
};

//landmark类型顶点，维度3维,类型是Eigen::Vector3d
class VertexPointBAL:public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  VertexPointBAL(){}
  
  //这里的读写功能函数就需要用了，参数分别是输入输出流类型实例的引用
  virtual bool read(std::istream& )
  {
    return false;
  }
  
  virtual bool write(std::ostream&) const
  {
    return false;
  }
  
  //设定顶点的初始值
  virtual void setToOriginImpl(){}
  
  virtual void oplusImpl(const double* update)
  {
    //将增量数组构造成增量矩阵
    //由于update是个double类型数组,而增量需要的是个矩阵
    //所以用update构造一个增量矩阵v,下面更新估计值,直接将v加上就好了
    Eigen::Vector3d::ConstMapType v(update);
    //将增量矩阵加到估计上
    _estimate += v;
  }
  
};

//BAL观测边,边即误差,继承自基础二元边，误差是重投影的像素误差
//参数为:误差维度2维,误差类型为Eigen::Vector2d,连接两个顶点VertexCameraBAL和VertexPointBAL
class EdgeObservationBAL:public g2o::BaseBinaryEdge<2,Eigen::Vector2d,VertexCameraBAL,VertexPointBAL>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeObservationBAL(){}
  
  //这里的读写功能函数就需要用了，参数分别是输入输出流类型实例的引用
  virtual bool read(std::istream& )
  {
    return false;
  }
  
  virtual bool write(std::ostream&) const
  {
    return false;
  }
  
  //误差计算函数
  //The virtual function comes from the Edge base class,Must define if you use edge
  virtual void computeError() override
  {
    //这里将第0个顶点,相机位姿取出来
    const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*>(vertex(0));
    //这里将第一个顶点,空间点位置取出来
    const VertexPointBAL* point = static_cast<const VertexPointBAL*>(vertex(1));
    
    //将相机位姿估计值,空间点位姿估计值,传给了重载的()运算符,这个重载,将计算好的结果输出到_error.data
    (*this)(cam->estimate().data(),point->estimate().data(),_error.data());
  }
  
  //这里即为重载的()函数,为模板函数,需要数据为相机位姿指针,空间点位置指针,用于承接输出误差的residuals
  //上面调用,用的_error.data()承接,完成误差计算
  //这个模板类还是用的重投影误差
  template<typename T>
  bool operator() (const T* camera,const T* point,T* residuals) const
  {
    //创建一个承接重投影像素坐标,也就是根据相机内外参和空间点坐标去投影得到的像素坐标,是估计值
    T predictions[2];
    
    //这个函数是投影过程
    CamProjectionWithDistortion(camera,point,predictions);
    
    //误差是估计值减去观测值
    residuals[0] = predictions[0] - T(measurement()(0));
    residuals[1] = predictions[1] - T(measurement()(1));
    
    return true;
  }
  
  //这里重写线性增量方程,也就是雅克比矩阵
  //求雅克比矩阵,之前都是用公式计算好的,直接对雅克比矩阵各个元素赋值,这里是利用ceres库中的
  //Autodiff去计算的，不管如何,要的结果就是要把雅克比矩阵_jacobianOplusXi,_jacobianOplusXj定义出来
  //这个函数就是求得误差关于优化变量的导数,由于这里误差跟两个优化变量有关，所以是二元边,所以求导就是
  //两个优化变量的偏导,就是jacobianOplusXi,jacobianOplusXj
  virtual void linearizeOplus() override
  {
    //使用数值求导
    //使用Ceres的自动求导,不然系统将调用g2o的数值求导
    
    //将相机顶点取出,赋值给cam
     //这里将第0个顶点,相机位姿取出来
    const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*>(vertex(0));
    //这里将第一个顶点,空间点位置取出来
    const VertexPointBAL* point = static_cast<const VertexPointBAL*>(vertex(1));
    
    //AutoDiff的定义是一个模板结构体,模板参数为代价函数类型,模板类型,代价函数的各参数维度(相机顶点维度,空间点维度)
    typedef ceres::internal::AutoDiff<EdgeObservationBAL,double,VertexCameraBAL::Dimension,VertexPointBAL::Dimension> BalAutoDiff;
    //定义一个行优先的double类型矩阵,大小为Dimension*VertexCameraBAL::Dimension,也就是2x9
    //这里就是误差对相机的导数
    Eigen::Matrix<double,Dimension,VertexCameraBAL::Dimension,Eigen::RowMajor> dError_dCamera;
    
     //定义一个行优先的double类型矩阵,大小为Dimension*VertexPointBAL::Dimension,也就是2x3
    //这里就是误差对空间点的导数
    Eigen::Matrix<double,Dimension,VertexPointBAL::Dimension,Eigen::RowMajor> dError_dPoint;
    
    //double*类型的数组,成员为double*,这里装了相机估计值数组指针和空间点估计值数组指针
    double* parameters[] = {const_cast<double*>(cam->estimate().data()),const_cast<double*>(point->estimate().data())};
    //雅克比矩阵为两块导数拼合起来的,一块是误差对相机的导数,一块是误差对空间点的导数
    double* jacobians[] = {dError_dCamera.data(),dError_dPoint.data()};
    
    //创建一个double类型的value数组,大小为Dimension，2个元素
    double value[Dimension];
    
    //这里就是一直所说的利用ceres的现行求导,这个Differentiate()就是在Auto结构体定义的
    //参数为1.代价函数,就是这个边类，直接用(*this)
    //2.参数列表,就是上面定义的有两个double指针的parameters数组，这两个指针一个指向相机参数数组,一个指向空间点数组
    //3.输出的维度,就是边的维度，2维
    //4.误差函数functor的输出值,也就是*this计算出来的误差
    //5.最终要求的雅克比矩阵
    bool diffState = BalAutoDiff::Differentiate(*this,parameters,Dimension,value,jacobians);
  
    //雅克比矩阵就计算完成了,最后就是赋值给_jacobianOplusXi和_jacobianOplusXj
    if(diffState)
    {
      _jacobianOplusXi = dError_dCamera;
      _jacobianOplusXj = dError_dPoint;
    }
    else
    {
      assert(0 && "Error while differentiating");
      _jacobianOplusXi.setZero();
      _jacobianOplusXj.setZero();
    }
  }
};