#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

void drawAxis(cv::Mat&, cv::Point, cv::Point, cv::Scalar, const float);
double getOrientation(const std::vector<cv::Point>&, cv::Mat&);

void drawAxis(cv::Mat& img, cv::Point p, cv::Point q, cv::Scalar colour, const float scale = 0.2)
{
    double angle = std::atan2((double)p.y - q.y, (double)p.x - q.x);
    double hypotenuse = std::sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));

    q.x = (int)(p.x - scale * hypotenuse * std::cos(angle));
    q.y = (int)(p.y - scale * hypotenuse * std::sin(angle));
    cv::line(img, p, q, colour, 1, cv::LINE_AA);

    p.x = (int)(q.x + 9 * std::cos(angle + CV_PI / 4));
    p.y = (int)(q.y + 9 * std::sin(angle + CV_PI / 4));
    cv::line(img, p, q, colour, 1, cv::LINE_AA);

    p.x = (int)(q.x + 9 * std::cos(angle - CV_PI / 4));
    p.y = (int)(q.y + 9 * std::sin(angle - CV_PI / 4));
    cv::line(img, p, q, colour, 1, cv::LINE_AA);
}

double getOrientation(const std::vector<cv::Point>& pts, cv::Mat& img)
{
    int sz = static_cast<int>(pts.size());
    cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);

    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Point cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)), static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    std::vector<cv::Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);

    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0), pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }

    cv::circle(img, cntr, 3, cv::Scalar(255, 0, 255), 2);

    cv::Point p1 = cntr + 0.02 * cv::Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    cv::Point p2 = cntr - 0.02 * cv::Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));

    drawAxis(img, cntr, p1, cv::Scalar(0, 255, 0), 1);
    drawAxis(img, cntr, p2, cv::Scalar(255, 255, 0), 5);

    double angle = std::atan2(eigen_vecs[0].y, eigen_vecs[0].x);

    return angle;
}

int main(int argc, char** argv)
{
    std::string imagePath = "../Images/face_1.jpg";
    cv::Mat src = cv::imread(imagePath);

    if(src.empty())
    {
        std::cout << "ERROR: Could not open image !" << std::endl;
        return EXIT_FAILURE;
    }

    cv::imshow("Input", src);

    while (true)
    {
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat bw;
    cv::threshold(gray, bw, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bw, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);

        if (area < 1e2 || 1e5 < area)
        {
            continue;
        }

        cv::drawContours(src, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), 2);
        getOrientation(contours[i], src);
    }

    cv::imshow("Output", src);

    while (true)
    {
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }

    return EXIT_SUCCESS;
}