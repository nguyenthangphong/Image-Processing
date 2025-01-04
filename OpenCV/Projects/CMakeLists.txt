cmake_minimum_required(VERSION 3.10)

project(Projects)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_subdirectory(Face_Detection)
add_subdirectory(Measure_Size_Object)
