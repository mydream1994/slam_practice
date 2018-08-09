#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <boost/iterator/iterator_concepts.hpp>

#include "DBoW3/DBoW3.h"

using namespace std;
using namespace cv;

/*****************************
 *根据data/目录下的十张图片训练字典
 *****************************/

int main(int argc,char** argv)
{
  //read the image
  cout<<"reading images......"<<endl;
  vector<Mat> images;
  for(int i=0;i<10;i++)
  {
    string path = "../../data/"+to_string(i+1)+".png";
    images.push_back(imread(path));
  }
  
  //detect ORB features
  cout<<"detecting ORB features ..."<<endl;
  Ptr<Feature2D> detector = ORB::create();
  vector<Mat> descriptors;
  for(Mat& image:images)
  {
    vector<KeyPoint> keypoints;
    Mat descriptor;
    detector->detectAndCompute(image,Mat(),keypoints,descriptor);
    descriptors.push_back(descriptor);
  }
  
  //create vocabulary
  cout<<"createing vocabulary ... "<<endl;
  DBoW3::Vocabulary vocab;
  vocab.create(descriptors);
  cout<<"vocabulary info: "<<vocab<<endl;
  vocab.save("vocabulary.yml.gz");
  cout<<"done"<<endl;
  return 0;
}