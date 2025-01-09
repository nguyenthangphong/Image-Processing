cmake_minimum_required(VERSION 3.10)

project(face_recognition)

find_package(OpenCV REQUIRED)

find_package(dlib REQUIRED)

add_executable(
    face_recognition
    src/main.cpp
    src/Helpers.cpp
    src/PreProcessing.cpp
    src/TanTriggs.cpp
    src/TrainModel.cpp
    src/VideoFaceDetector.cpp
)

target_include_directories(
    face_recognition
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/include
)

target_link_libraries(face_recognition ${OpenCV_LIBS} dlib::dlib)
