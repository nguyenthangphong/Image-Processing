#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser = cv::CommandLineParser(argc, argv, "{camera|0|Camera device number.}");

    /* Load the haarcasecade */
    std::string faceCascadeName = "../../Face_Detection/haarcascades/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier faceCascade = cv::CascadeClassifier(faceCascadeName);

    if (faceCascade.empty())
    {
        std::cout << "Error: Could not loading face cascade." << std::endl;
        return -1;
    }

    /* Read the camera stream */
    int cameraDevice = parser.get<int>("camera");

    /* Load the video */
    cv::VideoCapture capture;
    capture.open(cameraDevice);

    if (!capture.isOpened())
    {
        std::cerr << "Error: Could not loading video." << std::endl;
        return -1;
    }

    int frameCount = 0;
    double fps = 0.0;
    double startTime = cv::getTickCount();

    int width = 640;
    int height = 480;
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat src;

    while (true)
    {
        cv::Mat dst;

        capture >> src;

        if (src.empty())
        {
            std::cerr << "Error: Could not loading the image from the video." << std::endl;
            break;
        }

        /* Resize the image */
        cv::resize(src, dst, cv::Size(width / 5, height / 5));

        /* Convert the RGB to GRAY format */
        cv::Mat gray;
        cv::cvtColor(dst, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        /* Detect the face */
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces);

        for (auto& face : faces)
        {
            cv::rectangle(dst, face, cv::Scalar(255, 0, 255), 2);
        }

        /* Calculate the FPS */
        frameCount++;
        double elapsedTime = (cv::getTickCount() - startTime) / cv::getTickFrequency();

        if (elapsedTime > 1.0)
        {
            fps = frameCount / elapsedTime;
            startTime = cv::getTickCount();
            frameCount = 0;
        }

        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(dst, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Face Detection", dst);

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}