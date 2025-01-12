cmake_minimum_required(VERSION 3.10)

project(decompanding)

find_package(OpenCV REQUIRED)

add_executable(decompanding main.cpp)

target_include_directories(decompanding PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(decompanding PRIVATE ${OpenCV_LIBS})