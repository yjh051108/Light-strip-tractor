cmake_minimum_required(VERSION 3.10)
project(armor_detection)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(tiaocan tiaocan.cpp)
target_link_libraries(tiaocan ${OpenCV_LIBS})
target_include_directories(tiaocan PRIVATE ${OpenCV_INCLUDE_DIRS})
