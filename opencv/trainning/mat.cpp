#include <iostream>
#include <opencv2/opencv.hpp>

std::string showSizeImage(cv::Mat image);
void showImage(cv::Mat image);
void showMatrixImage(cv::Mat& image);

std::string showSizeImage(cv::Mat image)
{
    std::string log = std::to_string(image.rows) + "x" + std::to_string(image.cols) + "x" + std::to_string(image.channels());
    return log;
}

void showImage(cv::Mat image)
{
    cv::imshow("Image", image);

    while (true)
    {
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }
}

void showMatrixImage(cv::Mat& image)
{
    if (image.empty())
    {
        std::cerr << "Error: The image matrix is empty !" << std::endl;
        return;
    }

    int channels = image.channels();

    /* GRAY */
    if (channels == 1)
    {
        for (int row = 0; row < image.rows; ++row)
        {
            const uchar* ptr = image.ptr<uchar>(row);

            for (int col = 0; col < image.cols; ++col)
            {
                std::cout << static_cast<int>(ptr[col]) << " ";
            }

            std::cout << std::endl;
        }
    }
    /* BGR */
    else if (channels == 3)
    {
        for (int row = 0; row < image.rows; ++row)
        {
            const cv::Vec3b *ptr = image.ptr<cv::Vec3b>(row);

            for (int col = 0; col < image.cols; ++col)
            {
                std::cout << "("<< static_cast<int>(ptr[col][0]) << ", " << static_cast<int>(ptr[col][1]) << ", " << static_cast<int>(ptr[col][2]) << ") ";
            }

            std::cout << std::endl;
        }
    }
    else
    {
        std::cerr << "Unsupported image type with " << channels << " channels." << std::endl;
    }
}

int main(int argc, char** argv)
{
    /* Create just the header parts */
    cv::Mat A, C;

    A = cv::imread(argv[1]);

    cv::Mat gray;
    cv::cvtColor(A, gray, cv::COLOR_BGR2GRAY);
    cv::Mat grayNormalized = gray / 255.0;
    showMatrixImage(grayNormalized);
    
    /* Copy constructor */
    cv::Mat B(A);

    /* Assignment operator */
    C = A;

    /* Using rectangle */
    cv::Mat D(A, cv::Rect(10, 10, 100, 100));

    /* Using row and column boundaries */
    cv::Mat E = A(cv::Range::all(), cv::Range(1, 3));

    cv::Mat F = A.clone();

    cv::Mat G;
    A.copyTo(G);

    cv::Mat M(2, 2, CV_8UC3, cv::Scalar(0, 0, 255));

    return 0;
}