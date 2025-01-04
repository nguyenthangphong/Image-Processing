#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    cv::Mat source, gray, thresh, dest;
    double area, scaleFactor, size;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    /* Load image */
    source = cv::imread(argv[1]);

    if (source.empty())
    {
        std::cerr << "Error: Could not opening image from path " << argv[1] << std::endl;
        return -1; 
    }

    /* Convert to grayscale */
    cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);

    /* Separate the object from the background */
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    /* Find the contours of the object */
    cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    /* Loop through the contours and calculate the area of each object */
    for (auto& contour : contours)
    {
        /* Get the area of the object in pixels */
        area = cv::contourArea(contour);

        /* Draw a bounding box around each object */
        cv::Rect box = cv::boundingRect(contour);
        std::cout << "box : x = " << box.x << ", y = " << box.y << ", width = " << box.width << ", height = " << box.height << std::endl;
        cv::rectangle(source, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(0, 255, 0), 1);

        /* Display the area on the image */
        std::string areaText = std::to_string(static_cast<int>(area));
        cv::putText(source, areaText, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }

    /* Display the image with the contours drawn */
    cv::imshow("Measure Size of an Object", source);

    while (true)
    {
        /* Press Q or ESC to exit */
        if (cv::waitKey(30) == 'q' || cv::waitKey(30) == 27)
        {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}