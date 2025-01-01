cmake_minimum_required(VERSION 3.10)

project(Face_Detection)

find_package(OpenCV REQUIRED)

add_executable(Face_Detection main.cpp)

target_include_directories(Face_Detection PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(Face_Detection PRIVATE ${OpenCV_LIBS})