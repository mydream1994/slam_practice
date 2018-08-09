
#通过add_library指令构建动态库和静态库
add_library(myslam SHARED frame.cpp mappoint.cpp map.cpp camera.cpp config.cpp)

#此库文件需要链接上放定义好的第三方库, ${THIRD_PARTY_LIBS}路径参数
target_link_libraries(myslam ${THIRD_PARTY_LIBS})