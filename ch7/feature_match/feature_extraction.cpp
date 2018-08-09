#include <iostream>
#include <stdio.h>
#include <boost/iterator/iterator_concepts.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc,char ** argv)
{
  Mat img_1 = imread("../1.png",CV_LOAD_IMAGE_COLOR);//以RGB格式加载图像,是默认参数
  Mat img_2 = imread("../2.png",CV_LOAD_IMAGE_COLOR);
  
  //初始化
  vector<KeyPoint> keypoints_1,keypoints_2;
  Mat descriptors_1,descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();  //特征检测器
  Ptr<DescriptorExtractor> descriptor = ORB::create();  //特征描述
  //匹配特征点使用汉明距离
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  
  //第一步:检测Oriented FAST 角点位置
  detector->detect(img_1,keypoints_1);
  detector->detect(img_2,keypoints_2);
  
  //第二步:根据角点位置计算BRIEF描述子
  descriptor->compute(img_1,keypoints_1,descriptors_1);
  descriptor->compute(img_2,keypoints_2,descriptors_2);
  
  Mat outimg1;
  drawKeypoints(img_1,keypoints_1,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  imshow("ORB特征点",outimg1);
  
  //第三步:对两幅图像中的BRIEF描述子进行匹配,使用汉明距离
  vector<DMatch> matches;
  matcher->match(descriptors_1,descriptors_2,matches);
  
  //第四步:匹配点对筛选
  double min_dist = 10000,max_dist = 0;
  
  //找出所有匹配之间的最小距离和最大距离,即是最相似的和最不相似的两组点之间的距离
  for(int i=0;i<descriptors_1.rows;i++)
  {
    double dist = matches[i].distance;
    if(dist < min_dist)
      min_dist = dist;
    if(dist > max_dist)
      max_dist = dist;
  }
  
  printf("-----Max_dist : %f \n",max_dist);
  printf("-----Min_dist : %f \n",min_dist);
  
  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误
  //但有时最小距离会非常小,设置一个经验值30作为下限
  vector<DMatch> good_matches;
  for(int i=0;i<descriptors_1.rows;i++)
  {
    if(matches[i].distance <= max(2*min_dist,30.0))
    {
      good_matches.push_back(matches[i]);
    }
  }
  
  //第五步：绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_match);
  drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches,img_goodmatch);
  
  imshow("所有匹配点对",img_match);
  imshow("优化后匹配点对",img_goodmatch);
  waitKey(0);
  return 0;
}