#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, const char** argv)
{
    /* Load the haarcasecade */
    std::string faceCascadeName = "../../Face_Detection/haarcascades/haarcascade_frontalface_default.xml";
    cv::CascadeClassifier faceCascade = cv::CascadeClassifier(faceCascadeName);

    if (faceCascade.empty())
    {
        std::cout << "Error: Could not loading face cascade." << std::endl;
        return -1;
    }

    /* Load the image */
    cv::Mat frame = cv::imread(argv[1]);

    if (frame.empty())
    {
        std::cerr << "Error: Could not opening image from path " << argv[1] << std::endl;
        return -1; 
    }

    /* Convert RGB to GRAY format */
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    /* Detect the face */
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(gray, faces);
    std::cout << "Detected " << faces.size() << " faces." << std::endl;

    for (auto& face : faces)
    {
        cv::rectangle(frame, face, cv::Scalar(255, 0, 255), 2);
    }

    cv::imshow("Face Detection", frame);

    while (true)
    {
        if (cv::waitKey(10) == 'q')
        {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}