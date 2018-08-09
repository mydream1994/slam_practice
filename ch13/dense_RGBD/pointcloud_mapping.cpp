#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Geometry>
//格式化字符串 处理图像文件格式
#include <boost/format.hpp>
//点云数据处理
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
using namespace cv;

int main(int argc,char** argv)
{
  vector<Mat> colorImgs,depthImgs;   //彩色图和深度图
  //欧式变换矩阵使用Eigen::Isometry3d,实际是4x4矩阵
  //在标准容器vector<>中使用Eigen库成员,不加Eigen::aligned_allocator，会报错
  vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d> > poses;   //相机位姿
  
  ifstream fin("../data/pose.txt");
  if(!fin)
  {
    cerr<<"请在有post.txt的目录下运行此程序"<<endl;
    return 1;
  }
  
  for(int i=0;i<5;i++)
  {
    boost::format fmt("../data/%s/%d.%s");    //图像文件格式
    colorImgs.push_back(imread((fmt%"color"%(i+1)%"png").str()));  
    //深度图是16UC1的单通道图像
    depthImgs.push_back(imread((fmt%"depth"%(i+1)%"pgm").str(),-1)); //使用-1读取原始深度图像
    
    double data[7] = {0};
    for(auto& d:data)
      fin>>d;
    Eigen::Quaterniond q(data[6],data[3],data[4],data[5]);
    Eigen::Isometry3d T(q);
    T.pretranslate(Eigen::Vector3d(data[0],data[1],data[2]));
    poses.push_back(T);
  }
  
  //计算点云并拼接
  //相机内参
  double cx = 325.5;  //图像像素,原点平移
  double cy = 253.5;
  double fx = 518.0;   //焦距和缩放  
  double fy = 519.0;
  double depthScale = 1000;
  
   //定义点云使用的格式   这里用的是XYZRGB
  typedef pcl::PointXYZRGB PointT;  //点云中的点对象 位置和像素值
  typedef pcl::PointCloud<PointT> PointCloud;    //整个点云对象
  
  //新建一个点云
  //使用智能指针,创建一个空点云,这种指针用完会自动释放
  PointCloud::Ptr pointCloud(new PointCloud);
  for(int i=0;i<5;i++)
  {
    PointCloud::Ptr current(new PointCloud);
    cout<<"转换图像中:"<<i+1<<endl;
    Mat color = colorImgs[i];
    Mat depth = depthImgs[i];    //深度图像
    Eigen::Isometry3d T=poses[i];   //每个图像对应的摄像头位姿
    
    //对每个像素值对应的点 转换到现实世界
    for(int v=0;v<color.rows;v++)
      for(int u=0;u<color.cols;u++)
      {
	//v表示指向第v行 u表示指向第u个元素
	unsigned int d = depth.ptr<unsigned short> (v)[u];
	if(d == 0) 
	  continue;   //为0表示没有测量到
	if(d >= 7000)
	  continue;      //深度太大时不稳定
	 Eigen::Vector3d point;
	point[2] = double(d)/depthScale;
	point[0] = (u-cx)*point[2]/fx;
	point[1] = (v-cy)*point[2]/fy;
	Eigen::Vector3d pointWorld = T*point;
	
	PointT p;
	p.x = pointWorld[0];
	p.y = pointWorld[1];
	p.z = pointWorld[2];
	//step每一航的字节数
	p.b = color.data[v*color.step+u*color.channels()];
	p.g = color.data[v*color.step+u*color.channels()+1];
	p.r = color.data[v*color.step+u*color.channels()+2];
	current->points.push_back(p);
      }
      //depth filter and statistical removal
      //利用统计滤波器方法去除孤立点，该滤波器统计每个点
      //与它最近N个点的距离值的分布,去除距离均值过大的点
      //这样就保留了那些"粘在一起"的点,去掉了孤立的噪声点
      PointCloud::Ptr tmp(new PointCloud);
      pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
      statistical_filter.setMeanK(50);
      statistical_filter.setStddevMulThresh(1.0);
      statistical_filter.setInputCloud(current);
      statistical_filter.filter(*tmp);
      (*pointCloud) += *tmp;
  }
  
  pointCloud->is_dense = false;
  cout<<"点云共有"<<pointCloud->size()<<"个点"<<endl;
  
  //voxel filter
  //利用体素滤波器进行降采样,由于多个视角存在视野重叠，在重叠区域会存在大量的位置十分相近的点
  //这会占用许多内存空间,体素滤波保证在某个一定大小的立方体(体素)内仅有一个点
  //相当与对三维空间进行了降采样,从而节省了很多存储空间
  pcl::VoxelGrid<PointT> voxel_filter;
  //把分辨率调至0.01,表示每立方厘米有一个点
  voxel_filter.setLeafSize(0.01,0.01,0.01);   //resolution
  PointCloud::Ptr tmp(new PointCloud);
  voxel_filter.setInputCloud(pointCloud);
  voxel_filter.filter(*tmp);
  tmp->swap(*pointCloud);
  
  cout<<"滤波之后，点云共有"<<pointCloud->size()<<"个点"<<endl;
  
   pcl::io::savePCDFileBinary("map.pcd",*pointCloud);
  cout<<"ok"<<endl;
  return 0;
}