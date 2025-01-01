#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

int main(int argc, char** argv)
{
    cv::CascadeClassifier faceCascade, eyeCascade;
    cv::Mat sourceImage, grayImage;
    std::vector<cv::Rect> faces;

    faceCascade = cv::CascadeClassifier("../../Face_Detection/haarcascades/haarcascade_frontalface_default.xml");

    if (faceCascade.empty())
    {
        std::cerr << "Error: Could not opening file haarcascade_frontalface_default.xml." << std::endl;
        return -1;
    }

    eyeCascade = cv::CascadeClassifier("../../Face_Detection/haarcascades/haarcascade_eye.xml");

    if (eyeCascade.empty())
    {
        std::cerr << "Error: Could not opening file haarcascade_eye.xml." << std::endl;
        return -1;
    }

    sourceImage = cv::imread(argv[1]);

    if (sourceImage.empty())
    {
        std::cerr << "Error: Could not opening image from path " << argv[1] << std::endl;
        return -1; 
    }

    cv::cvtColor(sourceImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayImage, grayImage);

    faceCascade.detectMultiScale(grayImage, faces, 1.3, 5);

    for (const auto& face : faces)
    {
        cv::rectangle(sourceImage, face, cv::Scalar(255, 255, 0), 2);

        cv::Mat faceROI = grayImage(face);
        cv::Mat faceColorROI = sourceImage(face);

        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(faceROI, eyes);

        for (const auto& eye : eyes)
        {
            cv::rectangle(faceColorROI, eye, cv::Scalar(0, 127, 255), 2);
        }
    }

    cv::imshow("Face Detection", sourceImage);

    while (true)
    {
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}