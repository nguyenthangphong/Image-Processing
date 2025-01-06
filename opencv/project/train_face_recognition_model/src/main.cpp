#include <iostream>
#include "include/face_recognition.h"

int main(int argc, char** argv)
{
    std::string imagePath = argv[1];

    if (detectFaces(imagePath))
    {
        std::cout << "Faces detected and recognized successfully!" << std::endl;
    }
    else
    {
        std::cout << "No faces detected or recognition failed." << std::endl;
    }

    return 0;
}