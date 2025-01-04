cmake_minimum_required(VERSION 3.10)

project(head_shots)

find_package(OpenCV REQUIRED)

add_executable(head_shots main.cpp)

target_include_directories(head_shots PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(head_shots PRIVATE ${OpenCV_LIBS})