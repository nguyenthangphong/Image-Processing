#include <opencv2/opencv.hpp>
#include <iostream>

int main(void)
{
    /* Define the camera matrix */
    float fx = 800.0f;
    float fy = 800.0f;
    float cx = 640.0f;
    float cy = 480.0f;
    cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    /* Define the distortion coefficients */
    cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << 0, 0, 0, 0, 0);

    /* Define the 3D point in the world coordinate system */
    float x = 10.0f;
    float y = 20.0f;
    float z = 30.0f;
    std::vector<cv::Point3f> points3D = {cv::Point3f(x, y, z)};

    /* Define the rotation and translation vectors */
    cv::Mat rvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);
    cv::Mat tvec = (cv::Mat_<float>(3, 1) << 0, 0, 0);

    /* Map the 3D point to 2D point */
    cv::Mat points2D;
    cv::projectPoints(points3D, rvec, tvec, cameraMatrix, distCoeffs, points2D);

    /* Display the 2D point */
    std::cout << "2D point: \n" << points2D << std::endl;

    return 0;
}