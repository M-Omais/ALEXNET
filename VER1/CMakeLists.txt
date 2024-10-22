cmake_minimum_required(VERSION 3.10)
project(PROJECT)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(PROJECT model.cpp)
target_link_libraries(PROJECT ${OpenCV_LIBS})
