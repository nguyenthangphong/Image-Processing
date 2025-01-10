#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
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

std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img);

int main(int argc, const char **argv)
{
    try
    {
        if (argc != 2)
        {
            std::cout << "Run this example by invoking it like this: " << std::endl;
            std::cout << "   ./face_recognition /home/phong/work/dataset/image/test/images.jpg" << std::endl;
            std::cout << std::endl;
            std::cout << "You will also need to get the face landmarking model file as well as " << std::endl;
            std::cout << "the face recognition model file.  Download and then decompress these files from: " << std::endl;
            std::cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << std::endl;
            std::cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << std::endl;
            std::cout << std::endl;
            return 1;
        }

        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

        dlib::shape_predictor shapePredictor68FaceLandmarks;
        dlib::deserialize(shapePredictor68FaceLandmarksPath) >> shapePredictor68FaceLandmarks;

        model::anet_type dlibFaceRecognitionResnetModelV1;
        dlib::deserialize(dlibFaceRecognitionResnetModelV1Path) >> dlibFaceRecognitionResnetModelV1;

        dlib::matrix<dlib::rgb_pixel> img;
        dlib::load_image(img, argv[1]);
        dlib::image_window win(img);

        std::vector<dlib::matrix<dlib::rgb_pixel>> faces;

        for (auto face : detector(img))
        {
            auto shape = shapePredictor68FaceLandmarks(img, face);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(std::move(face_chip));
            win.add_overlay(face);
        }

        if (faces.size() == 0)
        {
            std::cout << "No faces found in image!" << std::endl;
            return 1;
        }

        std::vector<dlib::matrix<float, 0, 1>> face_descriptors = dlibFaceRecognitionResnetModelV1(faces);

        std::vector<dlib::sample_pair> edges;

        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
            for (size_t j = i; j < face_descriptors.size(); ++j)
            {
                if (dlib::length(face_descriptors[i] - face_descriptors[j]) < 0.6)
                {
                    edges.push_back(dlib::sample_pair(i, j));
                }
            }
        }

        std::vector<unsigned long> labels;
        const auto num_clusters = dlib::chinese_whispers(edges, labels);
        std::cout << "Number of people found in the image: " << num_clusters << std::endl;

        std::vector<dlib::image_window> win_clusters(num_clusters);

        for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
        {
            std::vector<dlib::matrix<dlib::rgb_pixel>> temp;

            for (size_t j = 0; j < labels.size(); ++j)
            {
                if (cluster_id == labels[j])
                {
                    temp.push_back(faces[j]);
                }
            }

            win_clusters[cluster_id].set_title("face cluster " + dlib::cast_to_string(cluster_id));
            win_clusters[cluster_id].set_image(dlib::tile_images(temp));
        }

        std::cout << "face descriptor for one face: " << trans(face_descriptors[0]) << std::endl;

        dlib::matrix<float,0,1> face_descriptor = dlib::mean(dlib::mat(dlibFaceRecognitionResnetModelV1(jitter_image(faces[0]))));
        std::cout << "jittered face descriptor for one face: " << dlib::trans(face_descriptor) << std::endl;

        std::cout << "hit enter to terminate" << std::endl;
        std::cin.get();
    }
    catch (std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}

std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img)
{
    thread_local dlib::rand rnd;
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops;

    for (int i = 0; i < 100; ++i)
    {
        crops.push_back(dlib::jitter_image(img, rnd));
    }

    return crops;
}