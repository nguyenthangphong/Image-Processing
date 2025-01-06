cmake_minimum_required(VERSION 3.10)

project(train_face_recognition_model)

# Find the OpenCV and Dlib packages
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

# Add include directories to the specific target (modern CMake)
add_executable(
    train_face_recognition_model
    src/main.cpp
    src/face_recognition.cpp
)

# Use target_include_directories instead of include_directories
target_include_directories(
    train_face_recognition_model
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/include
)

# Link OpenCV and Dlib libraries
target_link_libraries(
    train_face_recognition_model
    ${OpenCV_LIBS}
    dlib::dlib
)
