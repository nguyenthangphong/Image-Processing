#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, const char** argv)
{
    /* Initialize face detector and shape predictor */
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp;
    dlib::anet


    return 0;
}