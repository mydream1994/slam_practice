#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <algorithm>
#include <boost/timer.hpp>

namespace myslam 
{
VisualOdometry::VisualOdometry():
  state_(INITIALIZING),ref_(nullptr),curr_(nullptr),map_(new Map),
  num_lost_(0),num_inliers_(0),matcher_flann_(new cv::flann::LshIndexParams(5,10,2))
{
      num_of_features_ = Config::get<int>("number_of_features");
      scale_factor_ = Config::get<double>("scale_factor");
      level_pyramid_ = Config::get<int>("level_pyramid");
      match_ratio_ = Config::get<float>("match_ratio");
      max_num_lost_ = Config::get<float>("max_num_lost");
      min_inliers_ = Config::get<int>("min_inliers");
      key_frame_min_rot = Config::get<double>("keyframe_rotation");
      key_frame_min_trans = Config::get<double>("keyframe_translation");
      map_point_erase_ratio_ = Config::get<double>("map_point_eraes_ratio");
      orb_ = cv::ORB::create(num_of_features_,scale_factor_,level_pyramid_);
}

VisualOdometry::~VisualOdometry()
{
    
}

bool VisualOdometry::addFrame(Frame::Ptr frame)
{
  switch(state_)
  {
    case INITIALIZING:
    {
      state_ = OK;
      curr_ = ref_ = frame;
      //extract features from first frame
      extractKeyPoints();
      computeDescriptors();
      //compute the 3d position of features in ref frame
      //setRef3DPoints();
      addKeyFrame();     //the first frame is a key-frame
      break;
    }
    case OK:
    {
      curr_ = frame;
      curr_->T_c_w_ = ref_->T_c_w_;
      extractKeyPoints();
      computeDescriptors();
      featureMatching();
      poseEstimationPnP();
      if(checkEstimatedPose() == true)  //a good estimation
      {
	//计算当前帧与世界坐标系之间的变换矩阵
	curr_->T_c_w_ = T_c_w_estimated_;//T_c_w = T_c_r*T_r*w;
	//ref_ = curr_;
	//setRef3DPoints();
	optimizeMap();  //实时改变地图
	num_lost_ = 0;
	if(checkKeyFrame() == true) //is a key-frame(当两帧移动一定距离时,可被选取为关键帧)
	{
	  addKeyFrame();
	}
      }
      else  //bad estimation due to various reasons
      {
	num_lost_++;
	if(num_lost_>max_num_lost_)
	{
	  state_ = LOST;
	}
	return false;
      }
      break;
    }
    case LOST:
    {
      cout<<"vo has lost."<<endl;
      break;
    }
  }
  return true;
}

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect(curr_->color_,keypoints_curr_); //找出特征点
    cout<<"extract keypoints cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::computeDescriptors()
{
  boost::timer timer;
   orb_->compute(curr_->color_,keypoints_curr_,descriptors_curr_);
   cout<<"descriptor computation cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::featureMatching()
{ 
  boost::timer timer;
  vector<cv::DMatch> matches;
  //select the candidates in map
  Mat desp_map;
  vector<MapPoint::Ptr> candidate;  //候选入地图中的路标点
  for(auto& allpoints:map_->map_points_)   //遍历所有路标点
  {
    MapPoint::Ptr& p = allpoints.second;
    //check if p in curr frame image
    if(curr_->isInFrame(p->pos_))   //判断此路标点是否在当前帧的视野范围内
    {
      //add to candidate
      p->visible_times_++;   
      candidate.push_back(p);   //将合适的路标点放入地图中
      desp_map.push_back(p->descriptor_);
    }
  }

  matcher_flann_.match(desp_map,descriptors_curr_,matches);  //将地图中的特征点与当前帧匹配
  //select the best matches
  
  float min_dis = std::min_element(matches.begin(),matches.end(),
               [](const cv::DMatch& m1,const cv::DMatch& m2)
	       {return m1.distance<m2.distance;})->distance;
	      
  match_3dpts_.clear();
  match_2dkp_index_.clear();
  for(cv::DMatch& m:matches)
  {
    if(m.distance<max<float>(min_dis*match_ratio_,30.0))
    {
      match_3dpts_.push_back(candidate[m.queryIdx]);  //路标点中的匹配点(世界坐标)
      match_2dkp_index_.push_back(m.trainIdx);    //与路标点匹配的当前帧的特征点在keypoints中的索引
      
    }
  }
  cout<<"good matches:"<<match_3dpts_.size()<<endl;
  cout<<"match cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::setRef3DPoints()
{
    //select the features with depth measurement
  pts_3d_ref_.clear();
  descriptors_ref_ = Mat();
  for(size_t i=0;i<keypoints_curr_.size();i++)
  {
    double d = ref_->findDepth(keypoints_curr_[i]);
    if(d>0)
    {
      Vector3d p_cam = ref_->camera_->pixel2camera(
	Vector2d(keypoints_curr_[i].pt.x,keypoints_curr_[i].pt.y),d);
      pts_3d_ref_.push_back(cv::Point3f(p_cam(0,0),p_cam(1,0),p_cam(2,0)));
      descriptors_ref_.push_back(descriptors_curr_.row(i));
    }
  }
}

void VisualOdometry::poseEstimationPnP()
{
  //construct the 3d 2d observation
  vector<cv::Point3f>  pts3d;
  vector<cv::Point2f>  pts2d;
  
#if 0
  for(cv::DMatch m:feature_matches_)
  {
    pts3d.push_back(pts_3d_ref_[m.queryIdx]);
    pts2d.push_back(keypoints_curr_[m.trainIdx].pt);
  }
#endif

 for(int index:match_2dkp_index_)
 {
   pts2d.push_back(keypoints_curr_[index].pt);  //当前帧匹配好的像素点 
 }
 for(MapPoint::Ptr pt:match_3dpts_)
 {
    pts3d.push_back(pt->getPositionCV());  //世界坐标
 }
 
  Mat K=(cv::Mat_<double>(3,3)<<
    ref_->camera_->fx_,0,ref_->camera_->cx_,
    0,ref_->camera_->fy_,ref_->camera_->cy_,
    0,0,1
  );
  Mat rvec,tvec,inliers;
  //100是迭代次数,4.0是重投影误差,0.99是可信度,inlier是输出的有效特征点数(pnp求解相机位姿)
  cv::solvePnPRansac(pts3d,pts2d,K,Mat(),rvec,tvec,false,100,4.0,0.99,inliers);
  num_inliers_ = inliers.rows;
  cout<<"pnp inliers:"<<num_inliers_<<endl;
  //rvec为旋转向量,tvec为平移向量
  T_c_w_estimated_ = SE3(
    SO3(rvec.at<double>(0,0),rvec.at<double>(1,0),rvec.at<double>(2,0)),
    Vector3d(tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0))
  );
  
  //using bundle adjustment to optimize the pose
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
  Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
  Block* solver_ptr = new Block(linearSolver);
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  
  g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(g2o::SE3Quat(T_c_w_estimated_.rotation_matrix(),T_c_w_estimated_.translation()));
  optimizer.addVertex(pose);
  
  //edges
  for(int i=0;i<inliers.rows;i++)
  {
    int index = inliers.at<int>(i,0);  //找出当前的特征点对应的是向量的哪个索引
    //3D->2D projection
    EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
    edge->setId(i);
    edge->setVertex(0,pose);
    edge->camera_ = curr_->camera_.get();
    edge->point_ = Vector3d(pts3d[index].x,pts3d[index].y,pts3d[index].z);
    edge->setMeasurement(Vector2d(pts2d[index].x,pts2d[index].y));
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    //set the inlier map points
    match_3dpts_[index]->matched_times_++;
  }
  
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  
  T_c_w_estimated_ = SE3(pose->estimate().rotation(),pose->estimate().translation());
  
  cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

bool VisualOdometry::checkEstimatedPose()
{
  //check if the estimated pose is good 
  if(num_inliers_<min_inliers_)
  {
    cout<<"reject because inlier is too small:"<<num_inliers_<<endl;
    return false;
  }
  //if the motion is too large,it is probably wrong
  SE3 T_r_c = ref_->T_c_w_*T_c_w_estimated_.inverse();
  Sophus::Vector6d d = T_r_c.log();
  if(d.norm() > 5.0)
  {
    cout<<"reject because motion is too large:"<<d.norm()<<endl;
    return false;
  }
  return true;
}

bool VisualOdometry::checkKeyFrame()
{
  //d为SE3的李代数(6维向量,平移在前,旋转在后)
  SE3 T_r_c = ref_->T_c_w_*T_c_w_estimated_.inverse();
   Sophus::Vector6d d = T_r_c.log();
   Vector3d trans = d.head<3>();
   Vector3d rot = d.tail<3>();
   if(rot.norm()>key_frame_min_rot || trans.norm()>key_frame_min_trans)
   {
     return true;
  }
  return false;
}

void VisualOdometry::addKeyFrame()
{
   if(map_->keyframes_.empty())  //如果没有关键帧
   {
     //first key-frame ,add all 3d points into map
     for(size_t i=0;i<keypoints_curr_.size();i++)
     {
       double d = curr_->findDepth(keypoints_curr_[i]);
       if(d<0)
	 continue;
       Vector3d p_world = ref_->camera_->pixel2world(
	   Vector2d(keypoints_curr_[i].pt.x,keypoints_curr_[i].pt.y),curr_->T_c_w_,d );
       Vector3d n = p_world-ref_->getCamCenter();//getCamCenter相机的光心(0,0,0)在世界坐标系的坐标
       n.normalize();
       MapPoint::Ptr map_point = MapPoint::createMapPoint(
	 p_world,n,descriptors_curr_.row(i).clone(),curr_.get()
      );
       map_->insertMapPoint(map_point);
    }
  }
   map_->insertKeyFrame(curr_);  //把有变化的关键帧放入地图中
   ref_ = curr_;
}

void VisualOdometry::addMapPoints()
{
    //add the new map points into map
    vector<bool> matched(keypoints_curr_.size(),false);
    for(int index:match_2dkp_index_)  //查看有哪些特征点被匹配了
      matched[index] = true;    
    
    for(int i=0;i<keypoints_curr_.size();i++)
    {
      if(matched[i] == true)
	continue;
      double d = ref_->findDepth(keypoints_curr_[i]);
      if(d<0)
	continue;
     Vector3d p_world = ref_->camera_->pixel2world(
	   Vector2d(keypoints_curr_[i].pt.x,keypoints_curr_[i].pt.y),curr_->T_c_w_,d );
     Vector3d n = p_world - ref_->getCamCenter(); //getCamCenter相机的光心(0,0,0)在世界坐标系的坐标
     n.normalize();  //归一化
     MapPoint::Ptr map_point = MapPoint::createMapPoint(
       p_world,n,descriptors_curr_.row(i).clone(),curr_.get()
         );
     map_->insertMapPoint(map_point);  //将没有被匹配的陌生的路标点放入地图中
    }
}

//
void VisualOdometry::optimizeMap()
{
    //remove the hardly seen and on visible points
    for(auto iter=map_->map_points_.begin();iter!=map_->map_points_.end();)
    {
      if(!curr_->isInFrame(iter->second->pos_))  //把地图中不在当前帧视野中的路标点删除
      {
	iter = map_->map_points_.erase(iter);
	continue;
      }
      //匹配次数与可见次数之比,匹配率过低说明经常见但是没有几次匹配,应该是难识别的点,应该舍弃
      float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
      if(match_ratio<map_point_erase_ratio_)
      {
	iter = map_->map_points_.erase(iter);
	continue;
      }
      
      double angle = getViewAngle(curr_,iter->second);
      if(angle > M_PI/6.)  //说明该点变化的角度过大
      {
	iter = map_->map_points_.erase(iter);
	continue;
      }
      if(iter->second->good_ == false)
      {
	 //to do triangulate this point
      }
      iter++;
    }
    
    //一般情况是运动幅度过大了,与之前的帧没多少交集了,所以增加一下新点
    if(match_2dkp_index_.size()<100)   //当匹配点减少时,增加地图的新点
    {
      addMapPoints();
    }
    //如果地图的点过多了,需要释放一些点
    if(map_->map_points_.size() > 1000)
    {
      //map map is too large,remove some one
      map_point_erase_ratio_+=0.05;
    }
    else
    {
      map_point_erase_ratio_=0.1;
    }
    cout<<"map points: "<<map_->map_points_.size()<<endl;
}

//取得一个空间点在不同两个帧下的视角角度,返回的是double类型的角度值
double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point)
{
   //空间坐标点减去相机中心坐标,得到从相机指向空间点的向量
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize(); //单位化
    return acos(n.transpose()*point->norm_);
}


}