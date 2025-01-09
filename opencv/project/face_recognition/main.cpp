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
    if (argc < 2)
    {
        std::cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << std::endl;
        exit(1);
    }

    std::string output_folder = ".";

    if (argc == 3)
    {
        output_folder = std::string(argv[2]);
    }

    std::string fn_csv = std::string(argv[1]);

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    try
    {
        read_csv(fn_csv, images, labels);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << "Error: Could not open file \"" << fn_csv << "\". Reason: " << e.msg << std::endl;
        exit(1);
    }

    if (images.size() <= 1)
    {
        std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(cv::Error::StsError, error_message);
    }

    int height = images[0].rows;

    cv::Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

    cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
    model->train(images, labels);

    int predictedLabel = model->predict(testSample);

    std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    std::cout << result_message << std::endl;

    cv::Mat eigenvalues = model->getEigenValues();
    cv::Mat W = model->getEigenVectors();
    cv::Mat mean = model->getMean();

    if (argc == 2)
    {
        cv::imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    }
    else
    {
        cv::imwrite(cv::format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    }

    for (int i = 0; i < std::min(10, W.cols); i++)
    {
        std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        std::cout << msg << std::endl;

        cv::Mat ev = W.col(i).clone();
        cv::Mat grayscale = norm_0_255(ev.reshape(1, height));
        cv::Mat cgrayscale;

        cv::applyColorMap(grayscale, cgrayscale, cv::COLORMAP_JET);

        if(argc == 2)
        {
            cv::imshow(cv::format("eigenface_%d", i), cgrayscale);
        }
        else
        {
            cv::imwrite(cv::format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        }
    }

    for (int num_components = std::min(W.cols, 10); num_components < std::min(W.cols, 300); num_components+=15)
    {
        cv::Mat evs = cv::Mat(W, cv::Range::all(), cv::Range(0, num_components));
        cv::Mat projection = cv::LDA::subspaceProject(evs, mean, images[0].reshape(1,1));
        cv::Mat reconstruction = cv::LDA::subspaceReconstruct(evs, mean, projection);

        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

        if(argc == 2)
        {
            cv::imshow(cv::format("eigenface_reconstruction_%d", num_components), reconstruction);
        }
        else
        {
            cv::imwrite(cv::format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        }
    }

    if (argc == 2)
    {
        while (true)
        {
            if (cv::waitKey(10) == 'q')
            {
                break;
            }
        }

        cv::destroyAllWindows();
    }

    return 0;
}