#include <opencv2/opencv.hpp>
#include <iostream>

int main(void)
{
    cv::VideoCapture camera = cv::VideoCapture(1);

    if (!camera.isOpened())
    {
        std::cout << "Camera can not open." << std::endl;
        return -1;
    }

    cv::Mat frame;
    camera >> frame;

    if (frame.empty())
    {
        std::cout << "Failed to capture frame 1. Exiting..." << std::endl;
        return -1;
    }

    std::cout << "Press 'q' to close the camera." << std::endl;

    while (true)
    {
        camera >> frame;

        if (frame.empty())
        {
            std::cout << "Failed to capture frame 2. Exiting..." << std::endl;
            return -1;
        }

        cv::imshow("Show camera", frame);

        if (cv::waitKey(1) == static_cast<int>('q'))
        {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}