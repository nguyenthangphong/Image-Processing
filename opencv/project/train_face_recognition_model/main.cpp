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

int main(void)
{
    std::cout << "[INFO] Start processing faces..." << std::endl;

    try
    {
        /* Initialize the face detector and shape detector */
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

        /* Load shape_predictor_68_face_landmarks.dat */
        dlib::shape_predictor shapePredictor68FaceLandmarks;
        dlib::deserialize(shapePredictor68FaceLandmarksPath) >> shapePredictor68FaceLandmarks;
        std::cout << "[INFO] Load shape_predictor_68_face_landmarks.dat done..." << std::endl;

        /* Load dlib_face_recognition_resnet_model_v1.dat */
        model::anet_type dlibFaceRecognitionResnetModelV1;
        dlib::deserialize(dlibFaceRecognitionResnetModelV1Path) >> dlibFaceRecognitionResnetModelV1;
        std::cout << "[INFO] Load dlib_face_recognition_resnet_model_v1.dat done..." << std::endl;

        std::vector<dlib::matrix<float, 0, 1>> knownEncodings;
        std::vector<std::string> knownNames;

        std::string datasetPath = "/home/phong/work/dataset/image";
        int imageCount = 0;

        for (const auto &entry : std::filesystem::recursive_directory_iterator(datasetPath))
        {
            if (entry.is_regular_file())
            {
                std::string imagePath = entry.path().string();
                std::string name = std::filesystem::path(imagePath).parent_path().filename().string();

                std::cout << "[INFO] Processing image: " << imagePath << std::endl;

                /* Load the image */
                cv::Mat image = cv::imread(imagePath);

                if (image.empty())
                {
                    std::cerr << "[WARNING] Could not read image: " << imagePath << std::endl;
                    continue;
                }

                /* Convert image to RGB for Dlib */
                cv::Mat rgb;
                cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

                /* Detect faces */
                dlib::cv_image<dlib::bgr_pixel> cimg(rgb);
                std::vector<dlib::rectangle> faces = detector(cimg);

                for (auto face : faces)
                {
                    /* Get face landmarks */
                    dlib::full_object_detection shape = shapePredictor68FaceLandmarks(cimg, face);

                    /* Extract face embedding */
                    dlib::matrix<dlib::rgb_pixel> faceChip;
                    dlib::extract_image_chip(cimg, dlib::get_face_chip_details(shape, 150, 0.25), faceChip);

                    dlib::matrix<float, 0, 1> faceDescriptor = dlibFaceRecognitionResnetModelV1(faceChip);

                    /* Store encoding and name */
                    knownEncodings.push_back(faceDescriptor);
                    knownNames.push_back(name);
                }

                imageCount++;
            }
        }

        /* Serialize encodings and names */
        std::cout << "[INFO] Serializing encodings..." << std::endl;
        std::ofstream encodingFile("/home/phong/work/dataset/model/encodings.dat", std::ios::binary);

        if (!encodingFile)
        {
            std::cerr << "[ERROR] Could not open file for writing." << std::endl;
            return -1;
        }

        for (size_t i = 0; i < knownEncodings.size(); ++i)
        {
            encodingFile.write(reinterpret_cast<const char*>(&knownEncodings[i](0, 0)), sizeof(float) * knownEncodings[i].size());
            size_t nameLength = knownNames[i].size();
            encodingFile.write(reinterpret_cast<const char*>(&nameLength), sizeof(size_t));
            encodingFile.write(knownNames[i].c_str(), nameLength);
        }

        encodingFile.close();
        std::cout << "[INFO] Training completed successfully." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
