#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

static cv::Mat norm_0_255(cv::InputArray _src)
{
    cv::Mat src = _src.getMat();

    cv::Mat dst;

    switch(src.channels())
    {
        case 1:
            cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }

    return dst;
}

static void readCSV(const std::string& fileName, std::vector<cv::Mat>& images, std::vector<int>& labels, char separator = ';')
{
    std::ifstream file(fileName.c_str(), std::ifstream::in);

    if (!file)
    {
        std::string errorMessage = "No valid input file was given, please check the given file name .";
        CV_Error(cv::Error::StsBadArg, errorMessage);
    }

    std::string line, path, classLabel;

    while (std::getline(file, line))
    {
        std::stringstream liness(line);
        std::getline(liness, path, separator);
        std::getline(liness, classLabel);

        if (!path.empty() && !classLabel.empty())
        {
            images.push_back(cv::imread(path, 0));
            labels.push_back(atoi(classLabel.c_str()));
        }
    }
}

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <csv.ext> <output_folder>" << std::endl;
        exit(1);
    }

    std::string outputFolder = ".";

    if (argc == 3)
    {
        outputFolder = std::string(argv[2]);
    }

    std::string pathCSV = std::string(argv[1]);

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    try
    {
        readCSV(pathCSV, images, labels);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Error: Opening file \"" << pathCSV << "\". Reason: " << e.msg << std::endl;
        exit(1); 
    }

    if (images.size() <= 1)
    {
        std::string errorMessage = "This demo needs at least 2 images to work. Please add more images to your data set !";
        CV_Error(cv::Error::StsError, errorMessage);
    }

    int height = images[0].rows;
    
    cv::Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];

    images.pop_back();
    labels.pop_back();

    cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
    model->train(images, labels);

    int predictedLabel = model->predict(testSample);

    std::string resultMessage = cv::format("Predicted class = %d / Actual class = %d .", predictedLabel, testLabel);
    std::cout << resultMessage << std::endl;

    cv::Mat eigenValues = model->getEigenValues();
    cv::Mat eigenVectors = model->getEigenVectors();

    cv::Mat mean = model->getMean();

    if (argc == 2)
    {
        cv::imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    }
    else
    {
        cv::imwrite(cv::format("%s/mean.png", outputFolder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    }

    for (int i = 0; i < cv::min(10, eigenVectors.cols); i++)
    {
        std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenVectors.at<double>(i));
        std::cout << msg << std::endl;

        cv::Mat ev = eigenVectors.col(i).clone();
        cv::Mat grayScale = norm_0_255(ev.reshape(1, height));

        cv::Mat grayImage;
        cv::applyColorMap(grayScale, grayImage, cv::COLORMAP_JET);

        if(argc == 2)
        {
            cv::imshow(cv::format("eigenface_%d", i), grayImage);
        }
        else
        {
            cv::imwrite(cv::format("%s/eigenface_%d.png", outputFolder.c_str(), i), norm_0_255(grayImage));
        }
    }

    for(int numComponents = cv::min(eigenVectors.cols, 10); numComponents < cv::min(eigenVectors.cols, 300); numComponents += 15)
    {
        cv::Mat evs = cv::Mat(eigenVectors, cv::Range::all(), cv::Range(0, numComponents));
        cv::Mat projection = cv::LDA::subspaceProject(evs, mean, images[0].reshape(1,1));
        cv::Mat reconstruction = cv::LDA::subspaceReconstruct(evs, mean, projection);

        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

        if(argc == 2)
        {
            cv::imshow(cv::format("eigenface_reconstruction_%d", numComponents), reconstruction);
        }
        else
        {
            cv::imwrite(cv::format("%s/eigenface_reconstruction_%d.png", outputFolder.c_str(), numComponents), reconstruction);
        }
    }

    if(argc == 2)
    {
        cv::waitKey(0);
    }

    return 0;
}