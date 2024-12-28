#include <iostream>
#include <opencv2/opencv.hpp>

int main(void)
{
    std::string videoPath = "../Video/video_1.mp4";

    cv::VideoCapture video(videoPath);

    if (!video.isOpened())
    {
        std::cerr << "Could not open video !" << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    double fps = video.get(cv::CAP_PROP_FPS);

    std::cout << "Information video:" << std::endl;
    std::cout << "Size: " << frameWidth << " x " << frameHeight << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    cv::Mat frame;

    while (true)
    {
        video >> frame;

        if (frame.empty())
        {
            std::cout << "End video !" << std::endl;
            break;
        }

        cv::imshow("Video", frame);

        char c = static_cast<char>(cv::waitKey(25));

        if (c == 'q' || c == 27)
        {
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();

    return -1;
}
