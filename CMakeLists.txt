cmake_minimum_required(VERSION 3.10)
project(ArmorDetection)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "D:/Program/Opencv490/build")
# 查找OpenCV库
find_package(OpenCV REQUIRED)
# 添加OpenCV头文件路径
include_directories(
    "D:/Program/Opencv490/include"
    "D:/Program/Opencv490/include/opencv2"
)
# 添加OpenCV库路径
link_directories("D:/Program/Opencv490/lib")


# 创建静态库
add_library(armor_lib STATIC
    ArmorDetector.cpp
    ArmorDetector.h
    MotionPredictor.cpp
    MotionPredictor.h
    PointTracker.h
    MotionEstimator.cpp
)

# 添加可执行文件
add_executable(armor_detector
    foretracktor.cpp
    trackor.cpp
)

# 链接库
target_link_libraries(armor_detector
    ${OpenCV_LIBS}
    armor_lib
)

# 设置头文件搜索路径
target_include_directories(armor_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(armor_lib PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)