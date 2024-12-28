#include <iostream>
#include <opencv2/opencv.hpp>

int main(void)
{
    cv::Mat frame;

    frame = cv::imread("../Images/face_1.jpg");

    if (frame.empty())
    {
        std::cerr << "Could not open image !" << std::endl;
        return -1;
    }

    cv::imshow("Image", frame);

    while (true)
    {
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }

    return 0;
}