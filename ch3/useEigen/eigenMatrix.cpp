#include <iostream>
#include <ctime>

//Eigen部分
#include <Eigen/Core>
//稠密矩阵的代数运算(逆，特征值等)
#include <Eigen/Dense>

using namespace std;
#define MATRIX_SIZE  50

int main(int argc,char **argv)
{
  //Eigen中所有向量和矩阵都是Eigen::Matrix,它是一个模板类,它的前三个参数为:数据类型,行,列
  //声明一个2x3的float矩阵
  Eigen::Matrix<float,2,3> matrix_23;
  
  //Eigen通过typedef提供了许多内置类型,不过底层人是Eigen::Matrix
  //Vector3d 实质上是Eigen::Matrix<double,3,1>,即三维向量
  Eigen::Vector3d v_3d;
  //一样的
  Eigen::Matrix<float,3,1> vd_3d;
  
  //Matrix3d实质上是Eigen::Matrix<double,3,3>
  Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();  //初始化为0
  
  //如果不确定矩阵大小,可以使用动态大小的矩阵
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_dynamic;
  //更简单的
  Eigen::MatrixXd matrix_x;
  
  //下面对矩阵操作
  matrix_23 <<1,2,3,4,5,6;
  cout<<matrix_23<<endl;
  
  //用()访问矩阵中的元素
  for(int i=0;i<2;i++)
  {
    for(int j=0;j<3;j++)
      cout<<matrix_23(i,j)<<"\t";
    cout<<endl;
  }
  
  //矩阵和向量相乘(实际上人是矩阵和矩阵)
  v_3d << 3,2,1;
  vd_3d << 4,5,6;
  //但是在Eigen中不能混合两种不同类型的矩阵,比如double和float
  //应该显式转换
  Eigen::Matrix<double,2,1> result = matrix_23.cast<double>() * v_3d;
  cout<<result<<endl;
  
  Eigen::Matrix<float,2,1> result2 = matrix_23 * vd_3d;
  cout<<result2<<endl;
  
  matrix_33 = Eigen::Matrix3d::Random();   //随机数矩阵
  cout<<matrix_33<<endl<<endl;
  
  cout<<matrix_33.transpose()<<endl;    //转置
  cout<<matrix_33.sum()<<endl;         //各元素和
  cout<<matrix_33.trace()<<endl;      //迹
  cout<<10*matrix_33<<endl;   //数乘
  cout<<matrix_33.inverse()<<endl;   //逆
  cout<<matrix_33.determinant()<<endl;   //行列式
  
  //特征值
  //实对称矩阵可以保证对角化成功
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
  cout<<"Eigen value = \n"<<eigen_solver.eigenvalues()<<endl;
  cout<<"Eigen vectors = \n"<<eigen_solver.eigenvectors()<<endl;
  
  //解方程
  //求解matrix_NN * x = v_Nd 这个方程
  //N的大小在前边的宏里定义,它由随机数生成
  //直接求逆自然是最直接的,但是求逆运算量大
  Eigen::Matrix<double,MATRIX_SIZE,MATRIX_SIZE>  matrix_NN;
  matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
  Eigen::Matrix<double,MATRIX_SIZE,1> v_Nd;
  v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE,1);
  
  clock_t time_stt = clock();  //计时
  //直接求逆
  Eigen::Matrix<double,MATRIX_SIZE,1> x = matrix_NN.inverse()*v_Nd;
  cout<<"time use in normal inverse is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
  
  //通常用矩阵分解来求,例如QR分解,速度会快很多
  time_stt = clock();
  x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout<<"time use in QR decomposition is "<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
  
  return 0;
}
