cmake_minimum_required(VERSION 3.10)

project(face_detection_using_camera)

find_package(OpenCV REQUIRED)

add_executable(face_detection_using_camera main.cpp)

target_include_directories(face_detection_using_camera PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(face_detection_using_camera PRIVATE ${OpenCV_LIBS})