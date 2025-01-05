#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Error: Invalid syntax execution." << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat src = cv::imread(argv[1]);

    if (src.empty())
    {
        std::cerr << "Error: Could not read image." << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat dst;
    std::vector<cv::Mat> channels;

    /* Separate color channels */
    cv::split(src, channels);

    /* Calculate the mean value of each channel */
    cv::Scalar meanR = cv::mean(channels[2]);
    cv::Scalar meanG = cv::mean(channels[1]);
    cv::Scalar meanB = cv::mean(channels[0]);

    /* Calculate the mean of all channels */
    double meanGray = (meanR[0] + meanG[0] + meanB[0]) / 3.0;

    /* Adjust channels for white balance */
    channels[2] = channels[2] * (meanGray / meanR[0]);
    channels[1] = channels[1] * (meanGray / meanG[0]);
    channels[0] = channels[0] * (meanGray / meanB[0]);

    /* Merge channels into final image */
    cv::merge(channels, dst);

    /* Show source image */
    cv::imshow("Source Image", src);

    while (true)
    {
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    /* Show destination image */
    cv::imshow("Destination Image", dst);

    while (true)
    {
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    return EXIT_SUCCESS;
}