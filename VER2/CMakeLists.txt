cmake_minimum_required(VERSION 3.10)
project(PROJECT)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add include directories (where fixed.h is located)
include_directories(${CMAKE_SOURCE_DIR})  # Assuming fixed.h is in the same directory

# Add source files
add_executable(PROJECT model.cpp )  # Include fixed.cpp if present

# Link OpenCV libraries
target_link_libraries(PROJECT ${OpenCV_LIBS})
