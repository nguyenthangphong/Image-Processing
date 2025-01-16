cmake_minimum_required(VERSION 3.10)

project(mapping_coordinate_3d_2d)

find_package(OpenCV REQUIRED)

add_executable(mapping_coordinate_3d_2d main.cpp)

target_include_directories(mapping_coordinate_3d_2d PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(mapping_coordinate_3d_2d PRIVATE ${OpenCV_LIBS})