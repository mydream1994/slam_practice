#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <boost/concept_check.hpp>

using namespace std;
using namespace cv;

/*使用SVD以及非线性优化来求解ICP，本实验使用两个RGB-D图像,通过特征匹配获取两组3D点,
 最后用ICP计算它们的位姿变换,opencv目前没有计算两组带匹配点的ICP的方法,需要自己实现*/

void find_feature_matches(
  const Mat& img_1,const Mat& img_2,
  vector<KeyPoint>& keypoints_1,
  vector<KeyPoint>& keypoints_2,
  vector<DMatch>& matches );

//像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d& p,const Mat& K);

//ICP求解3D-3D点对 求解R，t使用线性代数奇异值分解
void pose_estimation_3d3d(
  const vector<Point3f>& pts1,
  const vector<Point3f>& pts2,
  Mat& R,Mat& t
);

//g2o_Bundleadjustment优化,计算旋转和平移
//ICP算法求解3D-3D点对的转换矩阵后使用图优化进行优化
void bundleAdjustment(
  const vector<Point3f>& pts1,
  const vector<Point3f>& pts2,
  Mat& R,Mat& t
);

//需自定义边类型,误差项g2o edge
//误差模型--曲线模型的边,模板参数:观测值维度(输入的参数维度),类型,连接顶点类型(优化变量系统定义好的顶点或者自定义的顶点)
//3D点-3D点对
class EdgeProjectXYZRGBDPoseOnly:public g2o::BaseUnaryEdge<3,Eigen::Vector3d,g2o::VertexSE3Expmap> //基础一元,边类型
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  //类成员有Eigen变量时需要显示加此句话,宏定义
  //直接赋值
  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d& point):_point(point){}
  //误差计算
  virtual void computeError()
  {
    //0号顶点为位姿 类型强转
    const g2o::VertexSE3Expmap *pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
   //观测值为p,测量值为p'
    //对观测值进行变换后,与测量值做差得到误差
    _error = _measurement - pose->estimate().map(_point);
  }
  //3d-3d自定义求解器
  virtual void linearizeOplus()
  {
    //0号顶点为位姿 类型强转
    g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
    g2o::SE3Quat T(pose->estimate()); //得到变换矩阵
    Eigen::Vector3d xyz_trans = T.map(_point);  //对点进行变换
    double x = xyz_trans[0];  //变换后的x,y,z
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    //3x6的雅克比矩阵 误差对应的导数  优化变量更新增量
    _jacobianOplu