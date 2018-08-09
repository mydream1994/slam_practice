#include <iostream>
#include <fstream>
#include <boost/concept_check.hpp>
#include "ceres/ceres.h"

#include "SnavelyReprojectionError.h"
#include "common/BALProblem.h"
#include "common/BundleParams.h"

using namespace ceres;

void SetLinearSolver(ceres::Solver::Options* options, const BundleParams& params)
{
  //linear solver选取
  CHECK(ceres::StringToLinearSolverType(params.linear_solver,&options->linear_solver_type));
  CHECK(ceres::StringToSparseLinearAlgebraLibraryType(params.sparse_linear_algebra_library,&options->sparse_linear_algebra_library_type));
  CHECK(ceres::StringToDenseLinearAlgebraLibraryType(params.dense_linear_algebra_library,&options->dense_linear_algebra_library_type));
  //options->num_linear_solver_threads = params.num_threads;
}

//消元顺序设置
void SetOrdering(BALProblem* bal_problem,ceres::Solver::Options* options,const BundleParams& params)
{
  const int num_points = bal_problem->num_points();
  const int point_block_size = bal_problem->point_block_size();
  double* points = bal_problem->mutable_points();
  
  const int num_cameras = bal_problem->num_cameras();
  const int camera_block_size = bal_problem->camera_block_size();
  double* cameras = bal_problem->mutable_cameras();
  
  //这里如果设置为自动,则直接return
//   if(params.ordering == "automatic")
//     return;
  
  //创建个ParameterBlockOrdering类型的对象,在下面按顺序把参数码进去,达到排序的目的
  ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;
  
  //The points come before the cameras
  for(int i=0;i<num_points;i++)
  {
    ordering->AddElementToGroup(points+point_block_size*i,0);
  }
  
  for(int i=0;i<num_cameras;i++)
  {
    ordering->AddElementToGroup(cameras+camera_block_size*i,1);
  }
  //设置消元顺序
  options->linear_solver_ordering.reset(ordering);
  
}

void SetMinmizerOptions(Solver::Options* options,const BundleParams& params)
{
  //最大迭代次数
  options->max_num_iterations = params.num_iterations;
  //标准输出端输出优化log日志
  options->minimizer_progress_to_stdout = true;
  //设置线程,加速雅克比矩阵计算
 // options->num_threads = params.num_threads;
  
  //下降策略选取
  CHECK(StringToTrustRegionStrategyType(params.trust_region_strategy,
				      &options->trust_region_strategy_type));
}

void SetSolverOptionFromFlags(BALProblem* bal_problem,const BundleParams& params,Solver::Options* options)
{
  //ceres的设置是比较简单的,定义个option对象,直接设置options就行
  SetMinmizerOptions(options,params);
  cout<<"ok1"<<endl;
  SetLinearSolver(options,params);
  cout<<"ok2"<<endl;
  SetOrdering(bal_problem,options,params);
}

/*
 * 构建问题,主要是将优化数据和传入problem
 * @param bal_problem数据
 * @param problem 要构建的优化问题
 * @param params 优化所需参数
 */

void BuildProblem(BALProblem* bal_problem,Problem* problem,const BundleParams& params)
{
  //读取路标和相机的维度
  const int point_block_size = bal_problem->point_block_size();
  const int camera_block_size = bal_problem->camera_block_size();
  //还有路标和相机的数据首位置
  double* points = bal_problem->mutable_points();
  double* cameras = bal_problem->mutable_cameras();
  
  const double* observations = bal_problem->observations();
  
  for(int i=0;i<bal_problem->num_observations();i++)
  {
    //定义problem->AddResidualBlock()函数中重要的cost_function
    CostFunction* cost_function = SnavelyReprojectionError::Create(observations[2*i],observations[2*i+1]);
    //定义problem->AddResidualBlock()函数中需要的Hunber核函数
    LossFunction* loss_function = params.robustify ? new HuberLoss(1.0) : NULL;
    //定义problem->AddResidualBlock()函数中需要的待估计参数
    double* camera = cameras+camera_block_size*bal_problem->camera_index()[i];
    double* point = points+point_block_size*bal_problem->point_index()[i];
    
    /*
     * cost_function:代价函数
     * loss_function:损失函数,就是核函数
     * 紧接着的数组就是待优化参数了,由于个参数维度不同,所以类型为double*,有几个就写几个,这里两个,camera,point
     */
    
    problem->AddResidualBlock(cost_function,loss_function,camera,point);
  }
}

void SolveProblem(const char* filename,const BundleParams& params)
{
  //同样一开始,用BALProblem类对数据进行处理
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
  
  //构造最小二乘问题,problem其实就是目标函数
  Problem problem;
  BuildProblem(&bal_problem,&problem,params);
  
  std::cout<<"the problem is successfully build.. "<<std::endl;
  //优化选项设置
  Solver::Options options;
  
  SetSolverOptionFromFlags(&bal_problem,params,&options);
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  
  //summary输出优化简报
  Solver::Summary summary;
  //优化,传入选项设置,目标函数,简报输出变量
  ceres::Solve(options,&problem,&summary);
  std::cout<<summary.FullReport()<<endl;
  
   if(!params.final_ply.empty())
  {
    bal_problem.WriteToPLYFile(params.final_ply);
  }
}

int main(int argc,char** argv)
{
  BundleParams params(argc,argv);
  
  google::InitGoogleLogging(argv[0]);
  std::cout<<params.input<<std::endl;
  if(params.input.empty())
  {
   cout<<"Usage:bundle_adjuster -input<path for dataset>";
    return 1;
  }
  
  //直接调用SolveProblem就好,传入数据和优化参数
  SolveProblem(params.input.c_str(),params);
  return 0;
}