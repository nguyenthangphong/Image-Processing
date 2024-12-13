#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <tuple>

int main(void)
{
    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;

    cv::VideoCapture videoCapture("../Video/video1.mp4");

    if (!videoCapture.isOpened())
    {
        std::cerr << "Khong the mo video !" << std::endl;
        return -1; 
    }

    cv::Mat frame;
    videoCapture >> frame;

    cv::Rect bbox = cv::selectROI("Tracking", frame, false, false);

    cv::Ptr<cv::TrackerMIL> tracker = cv::TrackerMIL::create();

    tracker->init(frame, bbox);

    double fps = videoCapture.get(cv::CAP_PROP_FPS);
    double startTime = cv::getTickCount();
    int frameCount = 0;

    while (true)
    {
        videoCapture >> frame;

        /* Fail using cv::Rect2d, so using cv::Rect */
        bool ok = tracker->update(frame, bbox);

        if (ok)
        {
            std::tuple<int, int> tuple1(bbox.x, bbox.y);
            std::tuple<int, int> tuple2(bbox.x + bbox.width, bbox.y + bbox.height);
            cv::Point p1(std::get<0>(tuple1), std::get<1>(tuple1));
            cv::Point p2(std::get<0>(tuple2), std::get<1>(tuple2));
            cv::rectangle(frame, p1, p2, cv::Scalar(255, 0, 0), 2, 1);
        }
        else
        {
            cv::putText(frame, "Tracking failure detected", cv::Point(50, 50), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
        }

        frameCount += 1;
        double elapsedTime = (cv::getTickCount() - startTime) / cv::getTickFrequency();

        if (elapsedTime > 0)
        {
            fps = frameCount / elapsedTime;
        }

        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(50, 50), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Tracking", frame);

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}