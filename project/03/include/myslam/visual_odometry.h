#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/camera.h"
#include "myslam/map.h"
#include <boost/concept_check.hpp>

namespace myslam
{
  class VisualOdometry
  {
  public:
    typedef shared_ptr<VisualOdometry> Ptr;
    //VO本身有若干种状态,初始化,正常,丢失
    enum VOState{
      INITIALIZING=-1,
      OK=0,
      LOST
    };
    
    VOState state_;   //current VO State
    Map::Ptr map_;     //map with all frames and map points
    Frame::Ptr ref_;   //reference frame
    Frame::Ptr curr_;   //current frame
    
    cv::Ptr<cv::ORB>  orb_;  //orb detector and computer
    vector<cv::Point3f> pts_3d_ref_;   //3d points in reference frame
    vector<cv::KeyPoint> keypoints_curr_;   //keypoints in current frame
    Mat      descriptors_curr_;     //descriptor in current frame
    Mat      descriptors_ref_;       //descriptor in reference frame
    vector<cv::DMatch>  feature_matches_;  
    
    SE3 T_c_r_estimated_;  //the estimated pose of current frame
    int num_inliers_;    //number of inlier features in icp
    int num_lost_;      //number of lost times
    //parameters
    int num_of_features_;     //number of features
    double scale_factor_;      //scale in image pyramid //图像金字塔的规模
    int level_pyramid_;      //number of pyramid levels //金字塔的数量水平
    float match_ratio_;   //ratio for selecting good matches 选择好的匹配的比例 最小距离的两倍
    int max_num_lost_;   //max number of continuous lost times
    int min_inliers_;   //minimum inliers  检测到的有效特征点数量
    
    //两个关键帧的最小旋转
    double key_frame_min_rot;   //minimal rotation of two key-frames
    double key_frame_min_trans;  //minimal translation of two key-frames
    
  public:
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame(Frame::Ptr frame);  //add a new frame
    
  protected:
    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void poseEstimationPnP();  
    void setRef3DPoints();
    
    void addKeyFrame();
    bool checkEstimatedPose();
    bool checkKeyFrame();
  };
}

#endif