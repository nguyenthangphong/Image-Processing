#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <sys/stat.h>

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

        int key = cv::waitKey(1);
        /* Q pressed */
        if (key == 'q')
        {
            std::cout << "Escape hit, closing..." << std::endl;
            break;
        }
        /* Enter pressed */
        else if (key == 13)
        {
            std::string path = argv[2];
            std::string datasetPath = path + "/" + name;

            /* Create directory */
            if (mkdir(datasetPath.c_str(), 0777) == 0)
            {
                std::cout << "Directory created: " << datasetPath << std::endl;
            }
            else
            {
                std::cerr << "Error: Could not create directory." << std::endl;
                break;
            }

            std::string imageName = datasetPath + "/image_" + std::to_string(countImage) + ".jpg";

            if (!cv::imwrite(imageName, frame))
            {
                std::cerr << "Error: Failed to save image " << imageName << std::endl;
            }
            else
            {
                std::cout << imageName << " written!" << std::endl;
            }

            countImage += 1;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}