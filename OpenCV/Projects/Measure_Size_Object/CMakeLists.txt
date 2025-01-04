cmake_minimum_required(VERSION 3.10)

project(Measure_Size_Object)

find_package(OpenCV REQUIRED)

add_executable(Measure_Size_Object main.cpp)

target_include_directories(Measure_Size_Object PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(Measure_Size_Object PRIVATE ${OpenCV_LIBS})