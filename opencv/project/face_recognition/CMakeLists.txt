cmake_minimum_required(VERSION 3.10)

project(face_recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_package(dlib REQUIRED)

add_executable(
    face_recognition
    main.cpp
)

target_include_directories(
    face_recognition
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(face_recognition ${OpenCV_LIBS} dlib::dlib)
