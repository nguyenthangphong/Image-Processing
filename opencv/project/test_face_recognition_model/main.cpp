#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>

namespace model
{
    template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
    using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

    template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
    using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

    template <int N, typename SUBNET>
    using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;

    template <int N, typename SUBNET>
    using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

    template <typename SUBNET>
    using alevel0 = ares_down<256, SUBNET>;

    template <typename SUBNET>
    using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;

    template <typename SUBNET>
    using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;

    template <typename SUBNET>
    using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;

    template <typename SUBNET>
    using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

    using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
                                                                  alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;
};

std::string dlibFaceRecognitionResnetModelV1Path = "/home/phong/work/dataset/model/dlib_face_recognition_resnet_model_v1.dat";
std::string shapePredictor68FaceLandmarksPath = "/home/phong/work/dataset/model/shape_predictor_68_face_landmarks.dat";

double euclidean_distance(const dlib::matrix<float, 0, 1> &a, const dlib::matrix<float, 0, 1> &b)
{
    return dlib::length(a - b);
}

int main(int argc, const char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <test_image_path>" << std::endl;
        return -1;
    }

    std::string testImagePath = argv[1];

    try
    {
        /* Load models */
        dlib::shape_predictor shapePredictor68FaceLandmarks;
        dlib::deserialize(shapePredictor68FaceLandmarksPath) >> shapePredictor68FaceLandmarks;

        model::anet_type dlibFaceRecognitionResnetModelV1;
        dlib::deserialize(dlibFaceRecognitionResnetModelV1Path) >> dlibFaceRecognitionResnetModelV1;

        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

        /* Load encodings */
        std::vector<dlib::matrix<float, 0, 1>> knownEncodings;
        std::vector<std::string> knownNames;

        std::ifstream encodingFile("/home/phong/work/dataset/model/encodings.dat", std::ios::binary);

        if (!encodingFile)
        {
            std::cerr << "[ERROR] Could not open encodings file." << std::endl;
            return -1;
        }

        while (encodingFile.peek() != EOF)
        {
            dlib::matrix<float, 0, 1> encoding(128, 1);
            encodingFile.read(reinterpret_cast<char *>(&encoding(0, 0)), sizeof(float) * 128);

            size_t nameLength;
            encodingFile.read(reinterpret_cast<char *>(&nameLength), sizeof(size_t));
            std::string name(nameLength, ' ');
            encodingFile.read(&name[0], nameLength);

            knownEncodings.push_back(encoding);
            knownNames.push_back(name);
        }

        encodingFile.close();

        /* Process test image */
        cv::Mat img = cv::imread(testImagePath);

        if (img.empty())
        {
            std::cerr << "[ERROR] Could not read test image." << std::endl;
            return -1;
        }

        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        dlib::cv_image<dlib::bgr_pixel> cimg(rgb);

        auto faces = detector(cimg);

        for (auto face : faces)
        {
            dlib::full_object_detection shape = shapePredictor68FaceLandmarks(cimg, face);
            dlib::matrix<dlib::rgb_pixel> faceChip;
            dlib::extract_image_chip(cimg, dlib::get_face_chip_details(shape, 150, 0.25), faceChip);
            dlib::matrix<float, 0, 1> testDescriptor = dlibFaceRecognitionResnetModelV1(faceChip);

            double minDistance = 1.0;
            double threshold = 0.6;
            std::string bestMatch = "Unknown";

            for (size_t i = 0; i < knownEncodings.size(); ++i)
            {
                double distance = euclidean_distance(testDescriptor, knownEncodings[i]);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    bestMatch = knownNames[i];
                }
            }

            if (minDistance < threshold)
            {
                std::cout << "[INFO] Match found: " << bestMatch << " (distance: " << minDistance << ")" << std::endl;
            }
            else
            {
                std::cout << "[INFO] No match found (distance: " << minDistance << ")" << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
