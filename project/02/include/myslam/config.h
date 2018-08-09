#ifndef CONFIG_H
#define CONFIG_H

#include "myslam/camera.h"
#include <boost/concept_check.hpp>
namespace myslam
{
  class Config
  {
  private:
    static std::shared_ptr<Config> config_;
    //在文件读取方面,使用opencv的Filestorage类,它可以读取一个YAML文件,且可以访问其中任意一个字段
    cv::FileStorage file_;  
    
    Config(){}
  public:
    ~Config();
    
    //set a new config file
    static void setParameterFile(const std::string& filename);
    
    //access the parameter values
    template<typename T>
    static T get(const std::string& key)
    {
      return T(Config::config_->file_[key]);
    }
  };
}
#endif