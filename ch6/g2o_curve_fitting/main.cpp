#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
#include <boost/concept_check.hpp>

using namespace std;

//曲线模型的顶点,模板参数:优化变量维度和数据类型
class CurveFittingVertex: public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual void setToOriginImpl()   //重置
  {
    _estimate << 0,0,0;
  }
  
  virtual void oplusImpl(const double* update) //更新
  {
    _estimate+= Eigen::Vector3d(update);
  }
  //存盘和读盘:留空
  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{}
};

//误差模型 模板参数: 观测值维度,类型,连接顶点类型
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CurveFittingEdge(double x): BaseUnaryEdge(),_x(x){}
  //计算曲线模型误差
  void computeError()
  {
    const CurveFittingVertex * v = static_cast<const CurveFittingVertex*>(_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    _error(0,0) = _measurement - exp(abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0));
  }
  
 //存盘和读盘:留空
  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{}
  
public:
  double _x;   //x值,y值为_measurement
};

int main()
{
  double a=1.0,b=2.0,c=1.0;    //真实参数值
  int N=100;            //数据点
  double w_sigma = 1.0;    //噪声sigma值(高斯分布的标准差)
  cv::RNG rng;            //opencv随机数生成器
  double abc[3] = {0,0,0};            //a,b,c参数的估计值
  
  vector<double> x_data,y_data;  //数据
  
  cout<<"generating data:"<<endl;
  for(int i=0;i<N;i++)
  {
    double x=i/100.0;
    x_data.push_back(x);
    y_data.push_back(exp(a*x*x + b*x + c)+rng.gaussian(w_sigma));
    
    cout<<x_data[i]<<" "<<y_data[i]<<endl;
  }
  
  //构建图优化,先设定g2o
  //每个误差项优化变量维度为3,误差值维度为1
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> Block;
  //线性方程求解器
  Block::LinearSolverType * linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
  Block * solver_ptr = new Block(linearSolver);    //矩阵块求解器
  //梯度下降法,从GN，LM，DogLeg中选
  g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  //g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
  //g2o::OptimizationAlgorithmDogleg * solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
 