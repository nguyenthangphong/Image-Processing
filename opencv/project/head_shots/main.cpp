#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, const char** argv)
{
    std::string name = argv[1];

    cv::VideoCapture capture = cv::VideoCapture(0);
    cv::namedWindow("Press space to take a photo", cv::WINDOW_NORMAL);
    cv::resizeWindow("Press space to take a photo", 768, 432);

    int countImage = 0;
    cv::Mat frame;

    while (true)
    {
        capture >> frame;

        if (frame.empty())
        {
            std::cerr << "Error: Could not loading the image." << std::endl;
            break;
        }

        cv::imshow("Press space to take a photo", frame);

        /* Esc pressed */
        if (cv::waitKey(1) == 27)
        {
            std::cout << "Escape hit, closing..." << std::endl;
            break;
        }
        /* Space pressed */
        else if (cv::waitKey(1) == 32)
        {
            std::string imageName = "../../dataset/" + name + "/image_" + std::to_string(countImage) + ".jpg";
            cv::imwrite(imageName, frame);
            std::cout << imageName << " written!" << std::endl;
            countImage += 1;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}