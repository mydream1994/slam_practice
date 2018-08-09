#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <boost/concept_check.hpp>

/*用opencv提供的EPnP求解PnP问题,然后通过g2o对结果进行优化
 由于PnP需要使用3D点,为了避免初始化带来的麻烦,使用了RGB-D相机中的深度图,
 作为特征点的3D位置*/

using namespace std;
using namespace cv;

void find_feature_matches(
  const Mat& img_1,const Mat& img_2,
  vector<KeyPoint>& keypoints_1,
  vector<KeyPoint>& keypoints_2,
  vector<DMatch>& matches );

//像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d& p,const Mat& K);

//这个bundle Adjustment问题,是一个最小化重投影误差的问题
//是一个非线性迭代优化问题
void bundleAdjustment(
  const vector<Point3f> points_3d,
  const vector<Point2f> points_2d,
  const Mat& K,
  Mat& R,Mat& t
);


int main(int argc,char **argv)
{
  //读取图像
  Mat img_1 = imread("../1.png",CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread("../2.png",CV_LOAD_IMAGE_COLOR);
  
  vector<KeyPoint> keypoints_1,keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
  cout<<"一共找到了"<<matches.size()<<"组匹配点"<<endl;
  
  //建立3D点
  //深度图为16位无符号数,单通道图像(CV_LOAD_IMAGE_UNCHANGED,载入原始图像包括alpha通道)
  Mat d1 = imread("../1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
  Mat K = (Mat_<double>(3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  
  for(DMatch m:matches)
  {
    ushort d = d1.ptr<unsigned short>(int (keypoints_1[m.queryIdx].pt.y))[int (keypoints_1[m.queryIdx].pt.x)];
    if( d==0 )
      continue;
    
    float dd = d/1000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt,K);
    pts_3d.push_back(Point3f(p1.x*dd,p1.y*dd,dd));  //图像特征点在相机坐标系下的3d坐标
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);  //在另一张图像上与之对应的特征点
  }
  
  cout<<"3d-2d pairs； "<<pts_3d.size()<<endl;
  
  Mat r,t;
   //调用opencv的pnp求解,可选择EPnp,DLS等方法
  solvePnP(pts_3d,pts_2d,K,Mat(),r,t,false);
  Mat R;
  Rodrigues(r,R);  //r为旋转向量形式,用Rodrigues公式转换为矩阵（opencv）
  
  cout<<"R= "<<endl<<R<<endl;
  cout<<"t= "<<endl<<t<<endl;
  /***上面求得的R，t是用pnp求得的，但还是会存在误差***/
  /**于是,我们将R,t作为初始值，进行bundleAdjustment优化***/
  cout<<"Calling bundle adjustment"<<endl;
  
  bundleAdjustment(pts_3d,pts_2d,K,R,t);
  return 0;
}

void find_feature_matches(
  const Mat& img_1,const Mat& img_2,
  vector<KeyPoint>& keypoints_1,
  vector<KeyPoint>& keypoints_2,
  vector<DMatch>& matches)
{
  //初始化
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
  //第三步:对两幅图像中的BRIEF描述子进行匹配,使用汉明距离
  vector<DMatch> match;
  matcher->match(descriptors_1,descriptors_2,match);
  
  //第四步:匹配点对筛选
  double min_dist = 10000,max_dist = 0;
  
  //找出所有匹配之间的最小距离和最大距离,即是最相似的和最不相似的两组点之间的距离
  for(int i=0;i<descriptors_1.rows;i++)
  {
    double dist = match[i].distance;
    if(dist < min_dist)
      min_dist = dist;
    if(dist > max_dist)
      max_dist = dist;
  }
  
  printf("-----Max_dist : %f \n",max_dist);
  printf("-----Min_dist : %f \n",min_dist);
  
  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误
  //但有时最小距离会非常小,设置一个经验值30作为下限
  for(int i=0;i<descriptors_1.rows;i++)
  {
    if(match[i].distance <= max(2*min_dist,30.0))
    {
      matches.push_back(match[i]);
    }
  }
}

//像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d& p,const Mat& K)
{
  return Point2d
	(
	  (p.x - K.at<double>(0,2))/K.at<double>(0,0),
	  (p.y - K.at<double>(1,2))/K.at<double>(1,1)
	);
}

void bundleAdjustment(
  const vector<Point3f> points_3d,
  const vector<Point2f> points_2d,
  const Mat& K,
  Mat& R,Mat& t
)
{
  //初始化g2o
  //构建图优化,先设定g2o
  //每个误差项优化变量维度为6(李代数为6维),landmark维度为3
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
  //线性方程求解器
  Block::LinearSolverType * linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
  Block * solver_ptr = new Block(linearSolver);    //矩阵块求解器
  //梯度下降法,从GN，LM，DogLeg中选
  g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  //g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
  //g2o::OptimizationAlgorithmDogleg * solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
  
  g2o::SparseOptimizer optimizer; //图模型
  optimizer.setAlgorithm(solver);   //设置求解器
  
  //节点:第二个相机的位姿节点,以及所有特征点的空间位置
  //边: 每个3D点在第二个相机中的投影
  //往图中增加顶点
  g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();  //相机位姿(李代数位姿)
  Eigen::Matrix3d R_mat;
  R_mat<<
	R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
	R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
	R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
  pose->setId(0);
  //使用g2o定义的相机位姿:SE3Quat,这个类内部使用了四元数加位移向量来存储位姿
  //同时也支持李代数上的运算
  pose->setEstimate(g2o::SE3Quat(R_mat,
				 Eigen::Vector3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0))
				));
  optimizer.addVertex(pose);
  
  //往图中添加路标节点
  int index=1;
  for(const Point3f p:points_3d) //points_3d表示在第一张图像的空间点(lanmarks)
  {
    //VertexSBAPointXYZ为空间点位置
    g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
    point->setId(index++);
    point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
    point->setMarginalized(true);   //g2o中必须设置marg，设置边缘化
    optimizer.addVertex(point);
  }
  
  //相机参数camera
  //类型g2o::CameraParameters,值为K.at<double>(0,0)和cx,cy组成的2维向量,0组成
  g2o::CameraParameters* camera = new g2o::CameraParameters(
    K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2),K.at<double>(1,2)),0
  );
  camera->setId(0);
  optimizer.addParameter(camera);
  
  //边
  index = 1;
  for(const Point2f p:points_2d)
  {
    //空间点的像素坐标,它的观测值为2维
    g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    edge->setId(index);
    //设置连接的顶点
    edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
    //设置顶点1为位姿
    edge->setVertex(1,pose);
    edge->setMeasurement(Eigen::Vector2d(p.x,p.y)); //观测数据
    edge->setParameterId(0,0);
    //信息矩阵2x2的单位阵
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }
  
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(100);   //可以指定优化步数
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout<<"optimization costs time :"<<time_used.count()<<" seconds. "<<endl;
  
  cout<<endl<<"after optimization:"<<endl;
  //欧式变换矩阵使用Eigen::Isometry3d(4x4)
  cout<<"T="<<endl<<Eigen::Isometry3d(pose->estimate()).matrix()<<endl;
}