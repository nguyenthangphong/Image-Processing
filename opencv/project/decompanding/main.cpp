#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

float decompanding(float x, const std::vector<cv::Point2f>& kneePoints)
{
    float y = 0.0f;

    for (size_t i = 0; i < kneePoints.size() - 1; ++i)
    {
        cv::Point2f p1 = kneePoints[i];
        cv::Point2f p2 = kneePoints[i + 1];

        if (x >= p1.x && x <= p2.x)
        {
            /* Linear interpolation between two break points */
            y = p1.y + (x - p1.x) * ((p2.y - p1.y) / (p2.x - p1.x));
            return y;
        }
    }

    return x;
}

void writeLogFile(cv::Mat src, std::string logFilePath)
{
    std::ofstream logFile(logFilePath);

    if (logFile.is_open())
    {
        logFile << src;
        logFile.close();
        std::cout << "Write " << logFilePath << " done..." << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not open log file!" << std::endl;
    }
}

int main(int argc, const char** argv)
{
    /* Create source matrix 8x8 */
    cv::Mat src = (cv::Mat_<uchar>(8, 8) << 
        0, 32, 64, 96, 128, 160, 192, 224,
        16, 48, 80, 112, 144, 176, 208, 240,
        8, 40, 72, 104, 136, 168, 200, 232,
        24, 56, 88, 120, 152, 184, 216, 248,
        4, 36, 68, 100, 132, 164, 196, 228,
        20, 52, 84, 116, 148, 180, 212, 244,
        12, 44, 76, 108, 140, 172, 204, 236,
        28, 60, 92, 124, 156, 188, 220, 252
    );

    /* Companding curve simulates square root */
    std::vector<cv::Point2f> kneePoints = {{0, 0}, {64, 40}, {128, 90}, {192, 160}, {255, 255}};

    /* Create destination matrix */
    cv::Mat dst = src.clone();

    /* Apply decompanding for each pixel */
    for (int x = 0; x < src.rows; ++x)
    {
        for (int y = 0; y < src.cols; ++y)
        {
            float pixelValue = static_cast<float>(src.at<uchar>(x, y));
            dst.at<uchar>(x, y) = static_cast<uchar>(decompanding(pixelValue, kneePoints));
        }
    }

    /* Write source matrix and destination matrix into log.txt */
    std::string srcLog = "source.txt";
    std::string dstLog = "destination.txt";

    writeLogFile(src, srcLog);
    writeLogFile(dst, dstLog);

    return 0;
}