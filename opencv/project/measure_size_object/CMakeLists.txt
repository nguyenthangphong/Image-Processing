cmake_minimum_required(VERSION 3.10)

project(measure_size_object)

find_package(OpenCV REQUIRED)

add_executable(measure_size_object main.cpp)

target_include_directories(measure_size_object PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(measure_size_object PRIVATE ${OpenCV_LIBS})