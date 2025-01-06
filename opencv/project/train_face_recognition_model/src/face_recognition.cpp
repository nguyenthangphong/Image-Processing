#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include "include/face_recognition.h"

/* define model */
namespace model
{
    /* define the layers used in the network */
    template <typename SUBNET>
    using conv1 = dlib::con<32, 7, 7, 2, 2, SUBNET>; /* first convolutional layer with 32 filters, 7x7 kernel, stride 2 */
    template <typename SUBNET>
    using relu1 = dlib::relu<SUBNET>; /* relu activation class */
    template <typename SUBNET>
    using fc1 = dlib::fc<128, SUBNET>; /* fully connected layer */
    template <typename SUBNET>
    using fc_no_bias1 = dlib::fc_no_bias<128, SUBNET>; /* fully connected layer not bias, output size 128 */

    /* define resnet face recognition network */
    using net_type = dlib::loss_metric<fc_no_bias1<dlib::avg_pool_everything<relu1<fc1<relu1<dlib::affine<conv1<dlib::input_rgb_image_sized<150>>>>>>>>>;
}

bool detectFaces(const std::string& imagePath)
{
    try
    {
        cv::Mat image = cv::imread(imagePath);

        if (image.empty())
        {
            std::cerr << "Error: Unable to load image!" << std::endl;
            return false;
        }

        /* convert the image from opencv to dlib */
        dlib::cv_image<dlib::bgr_pixel> dlibImage(image);

        /* load landmarks model */
        dlib::shape_predictor sp;
        dlib::deserialize("../../train_face_recognition_model/models/shape_predictor_68_face_landmarks.dat") >> sp;

        /* load face recognition model */
        model::net_type resnet;
        dlib::deserialize("../../train_face_recognition_model/models/dlib_face_recognition_resnet_model_v1.dat") >> resnet;

        /* detect the face */
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        std::vector<dlib::rectangle> faces = detector(dlibImage);

        if (faces.empty())
        {
            std::cout << "No faces detected." << std::endl;
            return false;
        }

        for (const auto& face : faces)
        {
            /* find the landmarks */
            dlib::full_object_detection shape = sp(dlibImage, face);

            /* convert the image to matrix for using network */
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(dlibImage, dlib::get_face_chip_details(shape), face_chip);

            /* calculate the face specifics */
            dlib::matrix<float, 0, 1> face_decriptor = resnet(face_chip);
            std::cout << "Face descriptor: " << dlib::trans(face_decriptor) << std::endl;
        }

        return true;
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}