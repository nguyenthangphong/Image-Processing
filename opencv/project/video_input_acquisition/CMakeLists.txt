cmake_minimum_required(VERSION 3.10)

project(video_input_acquisition)

find_package(OpenCV REQUIRED)

add_executable(video_input_acquisition main.cpp)

target_include_directories(video_input_acquisition PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(video_input_acquisition PRIVATE ${OpenCV_LIBS})