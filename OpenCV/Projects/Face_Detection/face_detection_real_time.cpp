#include <opencv2/opencv.hpp>
#include <iostream>

void detectAndDisplay(cv::Mat frame);
cv::CascadeClassifier faceCascade;
std::string faceCascadeName;

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser = cv::CommandLineParser(argc, argv, "{camera|0|Camera device number.}");

    /* Load the haarcasecade */
    faceCascadeName = "../../Face_Detection/haarcascades/haarcascade_frontalface_alt2.xml";

    if (!faceCascade.load(faceCascadeName))
    {
        std::cout << "Error: Could not loading face cascade." << std::endl;
        return -1;
    }

    /* Read the camera stream */
    int cameraDevice = parser.get<int>("camera");

    cv::VideoCapture capture;
    capture.open(cameraDevice);

    if (!capture.isOpened())
    {
        std::cout << "Error: Could not loading video capture." << std::endl;
        return -1;
    }

    cv::Mat frame;

    /* Output resize */
    int width = 640;
    int height = 480;
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            std::cout << "Error: No captured frame detected." << std::endl;
            break;
        }

        detectAndDisplay(frame);

        if (cv::waitKey(10) == 'q')
        {
            break;
        }
    }

    return 0;
}

void detectAndDisplay(cv::Mat frame)
{
    cv::Mat frameGray;
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frameGray, frameGray);

    /* Detect the face */
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(frameGray, faces);

    for (auto& face : faces)
    {
        cv::Mat faceROI = frameGray(face);
        int x = face.x;
        int y = face.y;
        int h = face.height;
        int w = face.width;
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(255, 0, 255), 2, 0, 8);
    }

    cv::imshow("Face Detection", frame);
}