#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>       //关于时间的库 C+11标准

using namespace std;

//代价函数的计算模型
struct CURVE_FITTING_COST
{
  CURVE_FITTING_COST(double x,double y):_x(x),_y(y){}
  
  //残差的计算
  template<typename T>
  //重载()运算符
  bool operator()(
          const T* const abc,  //模型参数,有三维
          T* residual  ) const   //残差       
  {
    residual[0] = T(_y) - ceres::exp(abc[0]*T(_x)*T(_x)+abc[1]*T(_x)+abc[2]);
    return true;
  }
  
  const double _x,_y;   //x,y数据
};

int main(int argc,char ** argv)
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
  
  //构造最小二乘问题
  ceres::Problem problem;
  for(int i=0;i<N;i++)
  {
    //调用AddResidualBlock将误差项添加到目标函数中
    problem.AddResidualBlock(
      //向问题中添加误差项
      //使用自动求导,模板参数: 误差类型,输出维度,输入维度,维度要与前面struct中一样
      //这里的误差是标量,维度为1,优化的是a,b,c三个量,维度为3
      new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(
	           new CURVE_FITTING_COST(x_data[i],y_data[i])
	),
	nullptr,         //核函数，这里不使用,为空
	abc              //待估计参数
    );
  }
  
  //配置求解器
  ceres::Solver::Options options;      //这里有很多配置项可以填
  options.linear_solver_type = ceres::DENSE_QR;  //增量方程如何求解(QR分解)
  options.minimizer_progress_to_stdout = true;     //输出到cout;
  
  ceres::Solver::Summary summary;     //优化信息
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  
  ceres::Solve (options,&problem,&summary);   //开始优化
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();  
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout<<"solve time coat ="<<time_used.count()<<" seconds. "<<endl;
  
  //输出结果
  cout<<summary.BriefReport()<<endl;
  cout<<"estimated a,b,c = ";
  for( auto a:abc)
    cout<<a<<" ";
  cout<<endl;
  return 0;
}