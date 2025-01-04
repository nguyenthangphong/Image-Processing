#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    cv::Mat src, dst;
    double threshold_value;

    src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    threshold_value = cv::threshold(src, dst, 127, 255, cv::THRESH_BINARY);
    std::cout << "Threshold value: " << threshold_value << std::endl;

    cv::imshow("Original Image", src);

    while (true)
    {
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }

    cv::imshow("Thresholded Image by Binary", dst);

    while (true)
    {
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }

    std::cout << "Source Image (src): " << src.rows << "x" << src.cols << std::endl;

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            std::cout << static_cast<int>(src.at<uchar>(i, j)) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Thresholded Image (dst):" << dst.rows << "x" << dst.cols << std::endl;

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            std::cout << static_cast<int>(dst.at<uchar>(i, j)) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}