#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    /* Organize the data set */
    std::vector<cv::Point2d> dataVector = {{150, 50}, {160, 55}, {170, 60}, {180, 65}, {190, 70}}; 

    int N = dataVector.size();
    cv::Mat dataMatrix = cv::Mat(N, 2, CV_64F);

    for (int i = 0; i < N; i++)
    {
        dataMatrix.at<double>(i, 0) = dataVector[i].x;
        dataMatrix.at<double>(i, 1) = dataVector[i].y;
    }

    /* Calculate the empirical mean */
    cv::Mat mean = cv::Mat::zeros(1, 2, CV_64F);
    
    for (int i = 0; i < dataMatrix.rows; i++)
    {
        mean.at<double>(0) += dataMatrix.at<double>(i, 0);
        mean.at<double>(1) += dataMatrix.at<double>(i, 1);
    }

    mean = mean / N;
    std::cout << "Mean:\n" << mean << std::endl;

    /* Calculate the deviations from the mean */
    cv::Mat deviation = cv::Mat(N, 2, CV_64F);

    for (int i = 0; i < N; i++)
    {
        deviation.at<double>(i, 0) = dataMatrix.at<double>(i, 0) - mean.at<double>(0);
        deviation.at<double>(i, 1) = dataMatrix.at<double>(i, 1) - mean.at<double>(1);
    }
    
    std::cout << "Deviation:\n" << deviation << std::endl;

    /* Find the covariance matrix */
    cv::Mat covarianceMatrix = cv::Mat::zeros(2, 2, CV_64F);

    for (int i = 0; i < N; i++)
    {
        covarianceMatrix.at<double>(0, 0) += deviation.at<double>(i, 0) * deviation.at<double>(i, 0);
        covarianceMatrix.at<double>(0, 1) += deviation.at<double>(i, 0) * deviation.at<double>(i, 1);
        covarianceMatrix.at<double>(1, 0) += deviation.at<double>(i, 1) * deviation.at<double>(i, 0);
        covarianceMatrix.at<double>(1, 1) += deviation.at<double>(i, 1) * deviation.at<double>(i, 1);
    }

    covarianceMatrix /= (N - 1);

    std::cout << "Covariance matrix:\n" << covarianceMatrix << std::endl;

    /* Find the eigenvectors and eigenvalues of the covariance matrix */
    cv::Mat eigenvalues = cv::Mat::zeros(2, 1, CV_64F);

    double covXX = covarianceMatrix.at<double>(0, 0);
    double covXY = covarianceMatrix.at<double>(0, 1);
    double covYX = covarianceMatrix.at<double>(1, 0);
    double covYY = covarianceMatrix.at<double>(1, 1);

    double a = 1.0;
    double b = -(covXX + covYY);
    double c = covXX * covYY - covYX * covXY;
    double discriminant = b * b - 4 * a * c;

    if (discriminant < 0)
    {
        std::cerr << "Error: No real eigenvalues ." << std::endl;
        return -1; 
    }

    eigenvalues.at<double>(0, 0) = (-b + std::sqrt(discriminant)) / (2 * a);
    eigenvalues.at<double>(1, 0) = (-b - std::sqrt(discriminant)) / (2 * a);

    std::cout << "Eigenvalues:\n" << eigenvalues << std::endl;

    cv::Mat eigenvectors = cv::Mat::zeros(2, 1, CV_64F);

    for (int i = 0; i < 2; i++)
    {
        
    }
    
    /* PCA */
    cv::PCA pca = cv::PCA(dataMatrix, cv::Mat(), cv::PCA::DATA_AS_ROW);
    std::cout << "Mean of PCA: \n" << pca.mean << std::endl;
    std::cout << "Eigenvalues of PCA: \n" << pca.eigenvalues << std::endl;
    std::cout << "Eigenvectors of PCA: \n" << pca.eigenvectors << std::endl;

    return 0;
}