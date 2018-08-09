#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main(int argc,char** argv)
{
  if(argc!=2)
  {
    cout<<"Usage: run_vo parameter_file"<<endl;
    return 1;
  }
  
  myslam::Config::setParameterFile(argv[1]);
  myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);  //构造函数会读取yaml文件中的配置内容
  
  string dataset_dir = myslam::Config::get<string>("dataset_dir"); //找出数据集路径
  cout<<"dataset:"<<dataset_dir<<endl;
  ifstream fin(dataset_dir+"/associate.txt");
  if(!fin)
  {
    cout<<"please generate the associate file called associate.txt!"<<endl;
    return 1;
  }
  
  vector<string> rgb_files,depth_files;
  vector<double> rgb_times,depth_times;
  while(!fin.eof()) //判断是否已读到文件尾,当读到最后一个字节时不会异常,下一次读不到数据会触发异常
  {
    string rgb_time,rgb_file,depth_time,depth_file;
    fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
    rgb_times.push_back(atof(rgb_time.c_str()));
    depth_times.push_back(atof(depth_time.c_str()));
    rgb_files.push_back(dataset_dir+"/"+rgb_file);
    depth_files.push_back(dataset_dir+"/"+depth_file);
    
    if(fin.good() == false)  //判断文件是否读到结尾
      break;
  }
  
  myslam::Camera::Ptr camera(new myslam::Camera);
  
  //visualization
  //可视化内容,用到opencv中的viz模块
  //定义3d可视化为vis,命名为Visual Odometry
  cv::viz::Viz3d vis("Visual Odometry");
  //定义世界坐标系和相机坐标系,构造参数是坐标系长度
  cv::viz::WCoordinateSystem world_coor(1.0),camera_coor(0.5);
  //设置视角
  //相机位置坐标,相机焦点坐标,相机y轴朝向
  cv::Point3d cam_pos(0,-1.0,-1.0),cam_focal_point(0,0,0),cam_y_dir(0,1,0);
  //由这三个参数,构造Affine3d类型的相机位姿,程序开始时的位姿
  cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos,cam_focal_point,cam_y_dir);;
  //设置可视化位姿为cam_pose
  vis.setViewerPose(cam_pose);
  
  //设置坐标系部件属性，然后添加到视图窗口
  //setRenderingProperty函数设置渲染属性,线宽
  world_coor.setRenderingProperty(cv::viz::LINE_WIDTH,2.0);
  camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH,1.0);
  //用showWidget函数将部件添加到窗口
  vis.showWidget("World",world_coor);
  vis.showWidget("Camera",camera_coor);
  //至此,窗口中已经显示需要显示的所有东西,就是两个坐标系:世界坐标系,相机坐标系
  //世界坐标系不动,需要做的是计算各帧的相机坐标系位置,不断给相机坐标系设置新的pose,达到动画的效果
  
  cout<<"read total"<<rgb_files.size()<<"entries"<<endl;
  for(int i=0;i<rgb_files.size();i++)
  {
    Mat color = cv::imread(rgb_files[i]);
    Mat depth = cv::imread(depth_files[i],-1);
    if(color.data == nullptr || depth.data == nullptr)
      break;
    
    myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
    pFrame->camera_ = camera;
    pFrame->color_ = color;
    pFrame->depth_ = depth;
    pFrame->time_stamp_ = rgb_times[i];
    
    boost::timer timer;
    vo->addFrame(pFrame);
    cout<<"VO cost time:"<<timer.elapsed()<<endl;
    
    if(vo->state_ == myslam::VisualOdometry::LOST)
    {
      break;
    }
    //求得是相机坐标系下的点在世界坐标系下的坐标
    SE3 Tcw = pFrame->T_c_w_.inverse(); //求逆(相机坐标系到世界坐标系)
    
    //show the map and the camera pose
    //用Tcw构造Affine3d类型的pose所需要的旋转矩阵和平移矩阵
    cv::Affine3d M(
      cv::Affine3d::Mat3(
	Tcw.rotation_matrix()(0,0),Tcw.rotation_matrix()(0,1),Tcw.rotation_matrix()(0,2),
	Tcw.rotation_matrix()(1,0),Tcw.rotation_matrix()(1,1),Tcw.rotation_matrix()(1,2),
	Tcw.rotation_matrix()(2,0),Tcw.rotation_matrix()(2,1),Tcw.rotation_matrix()(2,2)
      ),
      cv::Affine3d::Vec3(
	Tcw.translation()(0,0),Tcw.translation()(1,0),Tcw.translation()(2,0)
      )
    );
    
    cv::imshow("image",color);
    cv::waitKey(1);  //1ms
    //更新相机坐标系的位姿
    vis.setWidgetPose("Camera",M);
    vis.spinOnce(1,false);
  }
  return 0;
}
