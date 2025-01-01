#include <iostream>
#include <opencv2/opencv.hpp>

int main(void)
{
    cv::VideoCapture camera(0);

    if (!camera.isOpened())
    {
        std::cerr << "Could not open camera !" << std::endl;
        return -1;
    }

    cv::Mat frame;

    int frameCount = 0;
    double fps = 0.0;
    double startTime = cv::getTickCount();

    while (true)
    {
        camera >> frame;

        if (frame.empty())
        {
            std::cerr << "Could not get data from camera !" << std::endl;
            break;
        }

        frameCount++;
        double elapsedTime = (cv::getTickCount() - startTime) / cv::getTickFrequency();

        if (elapsedTime > 1.0)
        {
            fps = frameCount / elapsedTime;
            startTime = cv::getTickCount();
            frameCount = 0;
        }

        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Camera", frame);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}