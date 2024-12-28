#include <iostream>
#include <opencv2/opencv.hpp>

int main(void)
{
    cv::Mat frame;

    frame = cv::imread("Images/face_1.jpg");

    if (frame.empty())
    {
        std::cerr << "Cound not open image !" << std::endl;
        return -1;
    }

    cv::imshow("Image", frame);

    if (cv::waitKey(1) == 'q')
    {
        exit(1);
    }

    return 0;
}