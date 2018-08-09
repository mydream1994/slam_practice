#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
//Eigen 几何模块
#include <Eigen/Geometry>

/*本程序演示Eigen几何模块的使用方法*/

int main()
{
  //Eigen/Geomentry 模块提供了各种旋转和平移的表示
  //3D旋转矩阵直接使用Matrix3d 或 Matrix3f
  //单位矩阵
  Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
  cout<<rotation_matrix<<endl;
  //旋转向量使用AngleAxis 它底层不直接是Matrix,但运算可以当作矩阵(因为重载了运算符)
  //沿Z轴旋转45度
  Eigen::AngleAxisd rotation_vector(M_PI/4 , Eigen::Vector3d(0,0,1));
  cout.precision(3);   //输出小数点3位
  //用matrix()转换为矩阵
  cout<<"rotation  matrix=\n"<<rotation_vector.matrix()<<endl;
  //也可以直接赋值
  rotation_matrix = rotation_vector.toRotationMatrix();
  //用AngleAxis可以进行坐标变换
  Eigen::Vector3d v(1,0,0);
  Eigen::Vector3d v_rotated = rotation_vector*v;
  cout<<"(1,0,0) after rotation ="<<v_rotated.transpose()<<endl;
  
  //或者用旋转矩阵
  v_rotated = rotation_matrix*v;
  cout<<"(1,0,0) after rotation ="<<v_rotated.transpose()<<endl;
  
  //欧拉角:可以将旋转矩阵直接转换为欧拉角
  //ZYX顺序,即,yaw,pitch,roll
  Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);
  cout<<"yaw pitch roll="<<euler_angles.transpose()<<endl;
  //欧式变换矩阵使用Eigen::Isometry
  //虽然称为3d,实质上是4*4的矩阵
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  //按照rotation_vector进行旋转
  T.rotate(rotation_vector);
  //将平移向量设为(1,3,4)
  T.pretranslate(Eigen::Vector3d(1,3,4));
  cout<< " Transform matrix = \n"<<T.matrix()<<endl;
  
  //用变换矩阵进行坐标变换   相当于R*v + t
  Eigen::Vector3d v_transformed = T*v;
  cout<<"v tranformed = "<<v_transformed.transpose()<<endl;
  
  //四元数
  //直接把AngleAxis赋值给四元数,反之亦然
  Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
  //注意coeffs的顺序是(x,y,z,w),w是实部,前三个是虚部
  cout<<"quaterntion =\n"<<q.coeffs()<<endl;
  //也可以把旋转矩阵付给它
  q = Eigen::Quaterniond(rotation_matrix);
  cout<<"quaterntion =\n"<<q.coeffs()<<endl;
  
  //使用四元数旋转一个向量,使用重载的乘法即可
  v_rotated = q*v;   //注意数学上是qvq^{-1}
  cout<<"(1,0,0) after rotation ="<<v_rotated.transpose()<<endl;
  
  return 0;
}