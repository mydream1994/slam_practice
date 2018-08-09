#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h> 

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "common/BundleParams.h"
#include "common/BALProblem.h"
#include "g2o_bal_class.h"

using namespace Eigen;
using namespace std;
//Map类是矩阵库Eigen中用来将内存数据映射为任意形状的矩阵的类
typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
//给块求解器模板类定义维度并typedef,pose的维度为9维,landmark为3维
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9,3>>  BalBlockSolver;

//问题构建函数，传入一个BALProblem类型指针,稀疏求解器指针,参数类引用BundleParams&
void BuildProblem(const BALProblem* bal_problem,g2o::SparseOptimizer* optimizer,const BundleParams& params)
{
  //将bal_problem
  const int num_points = bal_problem->num_points();
  const int num_cameras = bal_problem->num_cameras();  //需要优化的相机变量有多少个
  const int camera_block_size = bal_problem->point_block_size(); //相机变量的维度
  const int point_block_size = bal_problem->point_block_size();
  
  //将相机数据的首位置读出,用于后方数据读取
  const double* raw_cameras = bal_problem->cameras();
  for(int i=0;i<num_cameras;++i)
  {
    //这里将9维相机位姿从数组中取出来构建成矩阵,用于下面的顶点的初始化赋值
    ConstVectorRef temVecCamera(raw_cameras+camera_block_size*i,camera_block_size);
    //开辟新的相机顶点类指针
    VertexCameraBAL* pCamera = new VertexCameraBAL();
    //设定初始值
    pCamera->setEstimate(temVecCamera);    
    //设定ID
    pCamera->setId(i);
    //rember to add vertex into optimizer
    optimizer->addVertex(pCamera);
  }
  
  //同样,这里将路标数据的首位置读出,用于后面读取
  const double* raw_points = bal_problem->points();
  for(int j=0;j<num_points;j++)
  {
    ConstVectorRef temVecPoint(raw_points+point_block_size*j,point_block_size);
    VertexPointBAL* pPoint = new VertexPointBAL();
    //设定初始值
    pPoint->setEstimate(temVecPoint);
    //设定ID，不能跟上面相机顶点重复,所以加了个相机个数,直接往后排
    pPoint->setId(j+num_cameras);
    
    //由于路标要被边缘化,所以设置边缘化属性为true
    pPoint->setMarginalized(true);
    optimizer->addVertex(pPoint);
  }
  
  //取出边的个数
  const int num_observations = bal_problem->num_observations();
  //取出边数组首位置
  const double* observations = bal_problem->observations();
  
  //用观测个数控制循环,来添加所有的边
  for(int i=0;i<num_observations;i++)
  {
    //开辟边内存指针
    EdgeObservationBAL* bal_edge = new EdgeObservationBAL();
    //由于每条边要定义连接的顶点ID，所以这里将cameraID和pointID取出
    const int camera_id = bal_problem->camera_index()[i]; //get id for the camera
    const int point_id = bal_problem->point_index()[i]+num_cameras;
    
    if(params.robustify)
    {
      g2o::RobustKernelHuber* rK = new g2o::RobustKernelHuber;
      rK->setDelta(1.0);
      bal_edge->setRobustKernel(rK);
    }
    
    //这里对每条边进行设置
    //连接的两个顶点
    bal_edge->setVertex(0,dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)));
    bal_edge->setVertex(1,dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)));
    //信息矩阵,依旧是单位阵
    bal_edge->setInformation(Eigen::Matrix2d::Identity());
    //设置默认值,就是将观测数据读进去
    bal_edge->setMeasurement(Eigen::Vector2d(observations[2*i],observations[2*i+1]));
    
    //将边添加进优化器
    optimizer->addEdge(bal_edge);
  }
}

//各个类的作用
//BALProblem跟优化数据txt对接,负责txt的读取,写入,同时还有生成PLY点云文件的功能
//BundleParams类负责优化需要的参数值,默认值设定和用户命令行输入等功能
//整体这样归类后,所有优化数据就去BALProblem类对象中询问,参数就去BundleParams类对象询问

//这个函数的作用是将优化后的结果再写入BALProblem类中
//在BALProblem类中,定义的所有读取写入功能都是BALProblem类与txt数据的,并没有优化后的数据与BALProblem
//所以这里定义了之后,就会产生优化后的数据类BALProblem,这样再跟txt或者PLY对接就很容易了
//参数：被写入的BALProblem*,优化器
void WriteToBALProblem(BALProblem* bal_problem,g2o::SparseOptimizer* optimizer)
{
   const int num_points = bal_problem->num_points();
  const int num_cameras = bal_problem->num_cameras();  //需要优化的相机变量有多少个
  const int camera_block_size = bal_problem->point_block_size(); //相机变量的维度
  const int point_block_size = bal_problem->point_block_size();
  
  //用mutable_cameras()函数取得相机首地址,用于后面数据写入
  double* raw_cameras = bal_problem->mutable_cameras();
  for(int i=0;i<num_cameras;i++)
  {
    VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
    Eigen::VectorXd NewCameraVec = pCamera->estimate();
    //取得估计值之后,就可以写入了
    memcpy(raw_cameras+i*camera_block_size,NewCameraVec.data(),sizeof(double)*camera_block_size);
    
  }
  
  double* raw_points = bal_problem->mutable_points();
  for(int j=0;j<num_points;++j)
  {
    VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(j+num_cameras));
    Eigen::Vector3d NewPointVec = pPoint->estimate();
    memcpy(raw_points+j*point_block_size,NewPointVec.data(),sizeof(double)*point_block_size);
    
  }
}

//求解设置:使用哪种下降方式,使用哪类线性求解器
/*
 * 设置求解选项,其实核心就是构建一个optimizer
 * @param bal_problem 优化数据
 * @param params 优化参数
 * @param optimizer 稀疏优化器
 */

void SetSolverOptionsFromFlags(BALProblem* bal_problem,const BundleParams& params,g2o::SparseOptimizer* optimizer)
{
  BalBlockSolver* solver_ptr;
  g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = nullptr;//线性方程求解器
  
  //使用稠密计算方法
  if(params.linear_solver == "dense_schur")
  {
    linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
  }
  //使用稀疏计算方法
  else if(params.linear_solver == "sparse_schur")
  {
    linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
    //让solver对矩阵排序保持稀疏性
    dynamic_cast<g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>*>(linearSolver)->setBlockOrdering(true);
  }
  
  //将线性求解器对象传入块求解器中,构造块求解器对象
  solver_ptr = new BalBlockSolver(linearSolver);
  
  //将块求解器对象传入下降策略中,构造下降策略对象
  g2o::OptimizationAlgorithmWithHessian* solver;
  //根据参数选择是LM还是DL
  if(params.trust_region_strategy == "levenberg_marquardt")
  {
    solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  }
  else if(params.trust_region_strategy == "dogleg")
  {
    solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
  }
  else   //没有输入下降策略或者输入错误时,输出报警并退出
  {
    cout<<"Please check your trust_region_strategy parameter again."<<endl;
    exit(EXIT_FAILURE);
  }
  
  //将下降策略传入优化器的优化逻辑中,至此,一个优化器就构建好了
  optimizer->setAlgorithm(solver);
}

//开始优化，这个优化函数参数就是待优化文件和优化参数
void SolveProblem(const char* filename,const BundleParams& params)
{
  BALProblem bal_problem(filename);
  cout<<"bal problem file loaded..."<<endl;
  
  cout<<"bal problem have "<<bal_problem.num_cameras()<<"cameras and"<<bal_problem.num_points()<<" points. "<<endl;
  //.num_observations()返回num_observations_值,显示观测边的数量
  cout<<"Forming "<<bal_problem.num_observations()<<" observations. "<<endl;
  
  if(!params.initial_ply.empty())
  {
    //优化前将BALProblem类中的数据生成一下点云数据,因为优化后,这个类中的数据会被覆盖
    bal_problem.WriteToPLYFile(params.initial_ply);
  }
  
  cout<<"beginning problem...."<<endl;
  
  //add some noise for the inital value
  srand(params.random_seed);
  bal_problem.Normalize();
  bal_problem.Perturb(params.rotation_sigma,params.translation_sigma,params.point_sigma);
  
  cout<<"Normalization Complete...."<<endl;
  
  //创建一个稀疏优化器对象
  g2o::SparseOptimizer optimizer;
  //用setSolverOptionsFromFlags()对优化器进行设置
  SetSolverOptionsFromFlags(&bal_problem,params,&optimizer);
  //设置完后,用BuildProblem()进行优化,参数：数据,优化器,参数
  BuildProblem(&bal_problem,&optimizer,params);
  
  cout<<"begin optimization .."<<endl;
  //开始优化
  optimizer.initializeOptimization();
  //输出优化信息
  optimizer.setVerbose(true);
  optimizer.optimize(params.num_iterations);
  
  cout<<"optimization Complete.. "<<endl;
  //优化完后,将优化的数据写入BALProblem类,此时这个类中原始数据已经被覆盖,在优化前,它已经生成过PLY点云文件
  WriteToBALProblem(&bal_problem,&optimizer);
  
  if(!params.final_ply.empty())
  {
    bal_problem.WriteToPLYFile(params.final_ply);
  }
}


int main(int argc,char** argv)
{
  //因为BundleParams类中自带了BA用的所有参数,并且都有默认值
  //由argc,argv构造也是类构造函数决定的,需要读一下命令行中有没有用户自定义的参数值,有读进来将默认值覆盖
  BundleParams params(argc,argv);
  
  //判断一下,如果输入的数据为空的话
  if(params.input.empty())
  {
    cout<<"Usage:bundle_adjuster -input<path for dataset>";
    return 1;
  }
  
  //直接调用SolveProblem就好,传入数据和优化参数
  SolveProblem(params.input.c_str(),params);
  return 0;
}