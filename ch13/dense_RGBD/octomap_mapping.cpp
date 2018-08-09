#include <iostream>
#include <fstream>

using namespace std;
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <Eigen/Geometry>
//格式化字符串 处理图像文件格式
#include <boost/format.hpp>

#include <octomap/octomap.h>  //for octomap

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
  
  cout<<"正在将图像转换为Octomap ... "<<endl;
  
  //octomap tree
  octomap::OcTree tree(0.05);    //参数为分辨率
  
  for(int i=0;i<5;i++)
  {
    
    cout<<"转换图像中:"<<i+1<<endl;
    Mat color = colorImgs[i];
    Mat depth = depthImgs[i];    //深度图像
    Eigen::Isometry3d T=poses[i];   //每个图像对应的摄像头位姿
    
    octomap::Pointcloud cloud;    //the point cloud in octomap
    
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
	//将世界坐标系的点放入点云
	cloud.push_back(pointWorld[0],pointWorld[1],pointWorld[2]);
      }
      
      //将点云存入八叉树地图,给定原点,这样可以计算投影线
      tree.insertPointCloud(cloud,octomap::point3d(T(0,3),T(1,3),T(2,3)));
  }
  
  //更新中间节点的占据信息并写入磁盘
  tree.updateInnerOccupancy();
  cout<<"saving octomap ... "<<endl;
  tree.writeBinary("octomap.bt");
  return 0;
}