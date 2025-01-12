#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, const char** argv)
{
    if (argc != 2)
    {
        std::cout << "Cách sử dụng: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    cv::VideoCapture capture = cv::VideoCapture(argv[1]);

    if (!capture.isOpened())
    {
        std::cerr << "Lỗi: Không thể mở video hoặc thiết bị !" << std::endl;
        return -1;
    }

    int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));

    int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    int fourcc = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));

    char codec[] =
    {
        static_cast<char>(fourcc & 0xFF),
        static_cast<char>((fourcc >> 8) & 0xFF),
        static_cast<char>((fourcc >> 16) & 0xFF),
        static_cast<char>((fourcc >> 24) & 0xFF),
        '\0'
    };

    int fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));

    int frame_count = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    int brightness = static_cast<int>(capture.get(cv::CAP_PROP_BRIGHTNESS));

    int contrast = static_cast<int>(capture.get(cv::CAP_PROP_CONTRAST));

    int saturation = static_cast<int>(capture.get(cv::CAP_PROP_SATURATION));

    std::cout << "Thông tin chi tiết video" << std::endl;
    std::cout << "Kích thước khung hình: " << width << " x " << height << std::endl;
    std::cout << "Codec: " << codec << std::endl;
    std::cout << "FPS: " << fps << " khung hình/s" << std::endl;
    std::cout << "Tổng số khung hình: " << frame_count << " khung hình" << std::endl;
    std::cout << "Tổng số thời gian: " << frame_count / fps << " s" << std::endl;
    std::cout << "Độ sáng: " << brightness << std::endl;
    std::cout << "Độ tương phản: " << contrast << std::endl;
    std::cout << "Độ bão hòa màu: " << saturation << std::endl;

    capture.release();

    return 0;
}