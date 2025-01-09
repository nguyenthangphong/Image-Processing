#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

static cv::Mat norm_0_255(cv::InputArray image)
{
    cv::Mat src = image.getMat();
    cv::Mat dst;
    int channel;

    channel = src.channels();

    switch(channel)
    {
        case 1:
            cv::normalize(image, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(image, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }

    return dst;
}

static void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);

    if (!file)
    {
        std::string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(cv::Error::StsBadArg, error_message);
    }

    std::string line, path, classlabel;

    while (std::getline(file, line))
    {
        std::stringstream liness(line);
        std::getline(liness, path, separator);
        std::getline(liness, classlabel);

        if(!path.empty() && !classlabel.empty())
        {
            images.push_back(cv::imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char** argv)
{
    if (argc < 3)
    {
        std::cout << "usage: " << argv[0] << " <train_csv.ext> <test_csv.ext>" << std::endl;
        exit(1);
    }

    std::string train_csv = std::string(argv[1]);
    std::string test_csv = std::string(argv[2]);

    std::vector<cv::Mat> train_images;
    std::vector<int> train_labels;

    try
    {
        read_csv(train_csv, train_images, train_labels);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << "Error: " << e.msg << std::endl;
        return -1;
    }

    if (train_images.size() <= 1)
    {
        std::cerr << "Training dataset must have at least 2 classes!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
    model->train(train_images, train_labels);

    std::vector<cv::Mat> test_images;
    std::vector<int> test_labels;

    try
    {
        read_csv(test_csv, test_images, test_labels);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << "Error: " << e.msg << std::endl;
        return -1;
    }

    for (size_t i = 0; i < test_images.size(); ++i)
    {
        cv::Mat testSample = test_images[i];
        int actualLabel = test_labels[i];

        cv::resize(testSample, testSample, train_images[0].size());

        int predictedLabel;
        double confidence;
        model->predict(testSample, predictedLabel, confidence);

        double threshold = 4000.0;

        if (confidence > threshold)
        {
            predictedLabel = -1;
        }

        std::cout << "Image " << i + 1 << ": Predicted = " << predictedLabel << ", Actual = " << actualLabel << ", Confidence = " << confidence << std::endl;
    }

    return 0;
}