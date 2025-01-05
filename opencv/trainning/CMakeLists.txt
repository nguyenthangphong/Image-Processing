cmake_minimum_required(VERSION 3.10)

project(trainning)

set(CMAKE_CXX_STANDARD 11)

find_package(trainning REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main pca.cpp)

target_link_libraries(main ${OpenCV_LIBS})
