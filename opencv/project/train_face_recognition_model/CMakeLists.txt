cmake_minimum_required(VERSION 3.10)

project(train_face_recognition_model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_package(dlib REQUIRED)

add_executable(
    train_face_recognition_model
    main.cpp
)

target_include_directories(
    train_face_recognition_model
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(train_face_recognition_model ${OpenCV_LIBS} dlib::dlib)
