#include <iostream>
#include <vector>
#include <fstream>
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>

#include <sophus/se3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using Sophus::SE3;
using namespace Eigen;

/************************
 * 演示了单目相机在已知轨迹下的稠密深度估计
 * 使用极线搜索+NCC匹配的方式
 ************************/

//parameters
const int boarder = 20;   //边缘宽度
const int width = 640;      //宽度
const int height = 480;     //高度
const double fx = 481.2f;   //相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 2;    //NCC取的窗口半宽度
const int ncc_area = (2*ncc_window_size+1)*(2*ncc_window_size+1);   //NCC窗口面积
const double min_cov = 0.1;   //收敛判定:最小方差
const double max_cov = 10;    //发散判定:最大方差



//从REMODE数据集读取数据
bool readDatasetFiles(
  const string& path,
  vector<string>& color_image_files,
  vector<SE3>& poses
);

//根据新的图像更新深度估计
bool update(
  const Mat& ref,
  const Mat& curr,
  const SE3& T_C_R,
  Mat& depth,
  Mat& depth_cov
);

//极线搜索
bool epipolarSearch(
  const Mat& ref,
  const Mat& curr,
  const SE3& T_C_R,
  const Vector2d& pt_ref,
  const double& depth_mu,
  const double& depth_cov,
  Vector2d& pt_curr
);

//更新深度滤波器
bool updateDepthFilter(
  const Vector2d& pt_ref,
  const Vector2d& pt_curr,
  const SE3& T_C_R,
  Mat& depth,
  Mat& depth_cov
);

//计算NCC评分
double NCC(const Mat& ref,const Mat& curr,const Vector2d& pt_ref,const Vector2d& pt_curr);

//双线性灰度插值
inline double getBilinearInterpolatedValue(const Mat& img,const Vector2d& pt)
{
  unsigned char* d = &img.data[int(pt(1,0))*img.step+int(pt(0,0))];  
  double xx = pt(0,0) - floor(pt(0,0));
  double yy = pt(1,0) - floor(pt(1,0));
  return((1-xx)*(1-yy)*double(d[0])+
        xx*(1-yy)*double(d[1])+
	(1-xx)*yy*double(d[img.step])+
	xx*yy*double(d[img.step+1]))/255.0;
}

//显示估计的深度图
bool plotDepth(const Mat& depth);

//像素到相机坐标系
inline Vector3d px2cam(const Vector2d px)
{
  return Vector3d(
    (px(0,0)-cx)/fx,
    (px(1,0)-cy)/fy,
     1
  );
}

//相机坐标系到像素
inline Vector2d cam2px(const Vector3d p_cam)
{
  return Vector2d(
    p_cam(0,0)*fx/p_cam(2,0)+cx,
    p_cam(1,0)*fy/p_cam(2,0)+cy
  );
}

//检测一个点是否在图像边框内
inline bool inside(const Vector2d& pt)
{
  return pt(0,0)>=boarder && pt(1,0)>=boarder
    && pt(0,0)+boarder<width && pt(1,0)+boarder<=height;
}

//显示极线匹配
void showEpipolarMatch(const Mat& ref,const Mat& curr,const Vector2d& pt_ref,const Vector2d& pt_curr);

//显示极线
void showEpipolarLine(const Mat& ref,const Mat& curr,const Vector2d& pt_ref,const Vector2d& px_min_curr,const Vector2d& px_max_curr);

int main(int argc,char** argv)
{
  if(argc != 2)
  {
    cout<<"Usage: dense_mapping path_to_test_dataset"<<endl;
    return -1;
  }
  
  //从数据集读取数据
  vector<string> color_image_files;
  vector<SE3> poses_TWC;
  bool ret = readDatasetFiles(argv[1],color_image_files,poses_TWC);
  if(ret == false)
  {
    cout<<"Reading image files failed!"<<endl;
    return -1;
  }
  
  cout<<"read total "<<color_image_files.size()<<" files. "<<endl;
  
  //第一张图
  Mat ref = imread(color_image_files[0],0);    //读取灰度图
  SE3  pose_ref_TWC = poses_TWC[0];
  double init_depth = 3.0;    //深度初始值
  double init_cov2 = 3.0;     //方差初始值
  Mat depth(height,width,CV_64F,init_depth);    //深度图
  Mat depth_cov(height,width,CV_64F,init_cov2);    //深度图方差
  
  for(int index=1;index<color_image_files.size();index++)
  {
    cout<<"**** loop "<<index<<" *** "<<endl;
    Mat curr = imread(color_image_files[index],0);
    if(curr.data == nullptr)
      continue;
    
    SE3 pose_curr_TWC = poses_TWC[index];
    SE3 pose_T_C_R = pose_curr_TWC.inverse()*pose_ref_TWC;  //坐标转换关系
    update(ref,curr,pose_T_C_R,depth,depth_cov);
    plotDepth(depth);
   // imshow("image",curr);
    waitKey(1);
  }
  
  cout<<"estimation returns, saving depth map ... "<<endl;
  imwrite("depth.png",depth);
  cout<<"done. "<<endl;
  return 0;
}


//从REMODE数据集读取数据
bool readDatasetFiles(
  const string& path,
  vector<string>& color_image_files,
  vector<SE3>& poses
)
{
  ifstream fin(path+"/first_200_frames_traj_over_table_input_sequence.txt");
  if(!fin)
  {
    return false;
  }
  
  while(!fin.eof())
  {
    //数据格式:图像文件名 tx,ty,tz,qx,qy,qz,qw,注意是TWC而非TCW
    string image;
    fin>>image;
    double data[7];
    for(double& d:data)
      fin>>d;
    
    color_image_files.push_back(path+string("/images/")+image);
    poses.push_back(
      SE3(Quaterniond(data[6],data[3],data[4],data[5]),
	Vector3d(data[0],data[1],data[2]))
    );
    if(!fin.good())
      break;
  }
  return true;
}

//对整个深度图进行更新
/************************************************************
 * 遍历当前图像的每一个像素,以第一张图像作为参考帧,对当前图像进行极线搜索,
 * 利用NCC搜索最佳匹配块,如果搜索成功,则更新深度图
 **********************************************************/
bool update(
  const Mat& ref,
  const Mat& curr,
  const SE3& T_C_R,
  Mat& depth,
  Mat& depth_cov
)
{
//并行程序设计(用于加速for循环处理速度)
#pragma omp parallel for
  for(int x=boarder;x<width-boarder;x++)
#pragma omp parallel for
    for(int y=boarder;y<height-boarder;y++)
    {
      //遍历每个像素
      //深度已收敛或发散
      if(depth_cov.ptr<double>(y)[x]<min_cov || depth_cov.ptr<double>(y)[x]>max_cov)
	continue;
      //在极线上搜索(x,y)的匹配
      Vector2d pt_curr;
      bool ret = epipolarSearch(
	   ref,
	   curr,
	   T_C_R,
	   Vector2d(x,y),
	   depth.ptr<double>(y)[x],
	   sqrt(depth_cov.ptr<double>(y)[x]),
	   pt_curr
      );
      
      if(ret == false)   //匹配失败
      {
	continue;
      }
      
      //显示匹配
      //showEpipolarMatch(ref,curr,Vector2d(x,y),pt_curr);
      
      //匹配成功,更新深度图
      updateDepthFilter(Vector2d(x,y),pt_curr,T_C_R,depth,depth_cov);
    }
}

//极线搜索
bool epipolarSearch(
  const Mat& ref, const Mat& curr,
  const SE3& T_C_R, const Vector2d& pt_ref,
  const double& depth_mu, const double& depth_cov,
  Vector2d& pt_curr
)
{
  Vector3d f_ref = px2cam(pt_ref);
  f_ref.normalize();  
  Vector3d P_ref = f_ref*depth_mu;  //参考帧的P向量
  
  Vector2d px_mean_curr = cam2px(T_C_R*P_ref);   //按深度均值投影的像素
  //假设深度值服从高斯分布,以均值为中心,左右各取正负3*depth_cov作为半径
  //然后在当前帧中寻找极线的投影
  double d_min = depth_mu-3*depth_cov,d_max = depth_mu+3*depth_cov;
  if(d_min<0.1)
    d_min = 0.1;
  
  Vector2d px_min_curr = cam2px(T_C_R*(f_ref*d_min));   //按最小深度投影的像素   
  Vector2d px_max_curr = cam2px(T_C_R*(f_ref*d_max));   //按最大深度投影的像素  
  
  Vector2d epipolar_line = px_max_curr - px_min_curr;   //极线(线段形式)
  Vector2d epipolar_direction = epipolar_line;    //极线方向
  epipolar_direction.normalize();  //归一化
  double half_length = 0.5*epipolar_line.norm();   //极线线段的半长度
  if(half_length > 100)
    half_length = 100;     //不希望搜索太多东西
  
  //显示极线
  // showEpipolarLine(ref,curr,pt_ref,px_min_curr,px_max_curr);
  
  //在极线上搜索,以深度均值为中心,左右各取半长度
  double best_ncc = -1.0;
  Vector2d best_px_curr;
  //遍历此极线上的像素(步长取0.7 = 根号2/2),寻找NCC最高的点作为匹配点,如果最高的NCC也低于阈值,匹配失败
  for(double l=-half_length;l<=half_length;l+=0.7)
  {
    Vector2d px_curr = px_mean_curr+l*epipolar_direction;  //待匹配的点
    if(!inside(px_curr))
    {
      continue;
    }
    //计算待匹配点与参考帧的NCC
    double ncc = NCC(ref,curr,pt_ref,px_curr);
    if(ncc > best_ncc)
    {
       best_ncc = ncc;
       best_px_curr = px_curr;
    }
  }
  
  if(best_ncc < 0.85f)     //如果最高的NCC也低于阈值,匹配失败
  {
    return false;
  }
  
  pt_curr = best_px_curr;
  return true;
}

//计算NCC评分(归一化互相关,计算的是两个小块的相关性)
double NCC(const Mat& ref,const Mat& curr,const Vector2d& pt_ref,const Vector2d& pt_curr)
{
  //零均值-归一化互相关
  //先算均值
  double mean_ref = 0,mean_curr = 0;
  vector<double> values_ref,values_curr;    //参考帧和当前帧的均值
  for(int x=-ncc_window_size;x<ncc_window_size;x++)
    for(int y=-ncc_window_size;y<ncc_window_size;y++)
    {
      double value_ref = double(ref.ptr<uchar>(int(y+pt_ref(1,0)))[int(x+pt_ref(0,0))])/255.0;
      mean_ref += value_ref;
      //双线性灰度插值
      double value_curr = getBilinearInterpolatedValue(curr,pt_curr+Vector2d(x,y));
      mean_curr += value_curr;
      
      values_ref.push_back(value_ref);
      values_curr.push_back(value_curr);
    }
    
  mean_ref /= ncc_area;
  mean_curr /= ncc_area;
  
  //计算Zero mean ncc_
  double numerator = 0,demoniator1 = 0,demoniator2 = 0;
  for(int i=0;i<values_ref.size();i++)
  {
    double n=(values_ref[i]-mean_ref)*(values_curr[i]-mean_curr);
    numerator += n;
    demoniator1 += (values_ref[i]-mean_ref)*(values_ref[i]-mean_ref);
    demoniator2 += (values_curr[i]-mean_curr)*(values_curr[i]-mean_curr);
  }
  
  return numerator / sqrt(demoniator1*demoniator2+1e-10);  //防止分母为零
}

//更新深度滤波器
/******************************
 * 三角化得到深度值
 * 利用上一节的内容计算深度的不确定性
 * 利用上一节的内容进行深度融合
 ******************************/
bool updateDepthFilter(
  const Vector2d& pt_ref,
  const Vector2d& pt_curr,
  const SE3& T_C_R,
  Mat& depth,
  Mat& depth_cov
)
{
  //用三角化计算深度
  SE3 T_R_C = T_C_R.inverse();
  Vector3d f_ref = px2cam(pt_ref);
  f_ref.normalize();
  Vector3d f_curr = px2cam(pt_curr);
  f_curr.normalize();
  
  //二阶方程用克莱默法则求解
  Vector3d t = T_R_C.translation(); 
  Vector3d f2 = T_R_C.rotation_matrix()*f_curr;
  Vector2d b = Vector2d(t.dot(f_ref),t.dot(f2));
  
  double A[4];
  A[0] = f_ref.dot(f_ref);
  A[2] = f_ref.dot(f2);
  A[1] = -A[2];
  A[3] = -f2.dot(f2);
  
  double d = A[0]*A[3] - A[1]*A[2];  //求矩阵的行列式
  
  Vector2d lambdavec = Vector2d(A[3]*b(0,0) - A[1]*b(1,0),
			      -A[2]*b(0,0) + A[0]*b(1,0))/d;
			      
  Vector3d xm = lambdavec(0,0)*f_ref;
  Vector3d xn = t+lambdavec(1,0)*f2;
  Vector3d d_esti = (xm+xn)/2.0;   //三角化算得的深度向量
  double depth_estimation = d_esti.norm(); //深度值
  
  //计算不确定性(以一个像素为误差)
  Vector3d p = f_ref*depth_estimation;
  Vector3d a = p-t;
  double t_norm = t.norm();
  double a_norm = a.norm();
  double alpha = acos(f_ref.dot(t)/t_norm);
  double beta = acos(-a.dot(t)/(a_norm*t_norm));
  double beta_prime = beta + atan(1/fx);
  double gamma = M_PI - alpha - beta_prime;
  double p_prime = t_norm*sin(beta_prime)/sin(gamma);
  double d_cov = p_prime - depth_estimation;   //方差
  double d_cov2 = d_cov*d_cov;
  
  //高斯融合
  double mu = depth.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))];
  double sigma2 = depth_cov.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))];
  
  double mu_fuse = (d_cov2*mu + sigma2*depth_estimation)/(sigma2+d_cov2);
  double sigma_fuse2 = (sigma2*d_cov2)/(sigma2+d_cov2);
  
  depth.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))] = mu_fuse;
  depth_cov.ptr<double>(int(pt_ref(1,0)))[int(pt_ref(0,0))] = sigma_fuse2;
  
  return true;
}

bool plotDepth(const Mat& depth)
{
     imshow("depth",depth*0.4);
     waitKey(1);
}

void showEpipolarMatch(const Mat& ref, const Mat& curr, const Vector2d& pt_ref, const Vector2d& pt_curr)
{
   Mat ref_show,curr_show;
   cvtColor(ref,ref_show,CV_GRAY2BGR);
   cvtColor(curr,curr_show,CV_GRAY2BGR);
   
   circle(ref_show,Point2f(pt_ref(0,0),pt_ref(1,0)),5,Scalar(0,0,250),2);
   circle(curr_show,Point2f(pt_curr(0,0),pt_curr(1,0)),5,Scalar(0,0,250),2);
   
   imshow("ref",ref_show);
   imshow("curr",curr_show);
   waitKey(1);
}

void showEpipolarLine(const Mat& ref, const Mat& curr, const Vector2d& pt_ref, const Vector2d& px_min_curr, const Vector2d& px_max_curr)
{
  Mat ref_show,curr_show;
  cvtColor(ref,ref_show,CV_GRAY2BGR);
  cvtColor(curr,curr_show,CV_GRAY2BGR);
  
  circle(ref_show,Point2f(pt_ref(0,0),pt_ref(1,0)),5,Scalar(0,255,0),2);
  circle(curr_show,Point2f(px_min_curr(0,0),px_min_curr(1,0)),5,Scalar(0,255,0),2);
  circle(curr_show,Point2f(px_max_curr(0,0),px_max_curr(1,0)),5,Scalar(0,255,0),2);
  line(curr_show,Point2f(px_min_curr(0,0),px_min_curr(1,0)),Point2f(px_max_curr(0,0),px_max_curr(1,0)),Scalar(0,255,0),1);

  imshow("ref",ref_show);
  imshow("curr",curr_show);
  waitKey(1);
}

