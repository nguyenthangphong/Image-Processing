# Choose version opencv
opencv_version = 4.10.0-dev
# Install OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x
# Install OpenCV-Contrib
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.x
# Build OpenCV Source
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D PYTHON_EXECUTABLE=/home/phong/anaconda3/envs/opencv-py/bin/python3 \
-D PYTHON_INCLUDE_DIR=/home/phong/anaconda3/envs/opencv-py/include/python3.7m \
-D PYTHON_LIBRARY=/home/phong/anaconda3/envs/opencv-py/lib/libpython3.7m.so \
-D OPENCV_PYTHON3_INSTALL_PATH=/home/phong/anaconda3/envs/opencv-py/lib/python3.7/site-packages \
-D WITH_QT=ON \
-DOPENCV_GENERATE_PKGCONFIG=ON \
-D WITH_OPENGL=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D BUILD_EXAMPLES=ON ..
make -j4
sudo make install
# Set up enviroment for python using anaconda
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
shasum -a 256 ~/Anaconda3-2024.10-1-Linux-x86_64.sh
./Anaconda3-2024.10-1-Linux-x86_64.sh
source ~/.bashrc
# Check install anaconda
anaconda-navigator
# Switch environment python
conda env list
conda activate opencv-py
conda deactivate
# Build
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/lib/pkgconfig:$PKG_CONFIG_PATH

# opencv version 4.x
g++ test_opencv.cpp -o test_opencv `pkg-config --cflags --libs opencv4`
# opencv version 3.x
g++ test_opencv.cpp -o test_opencv `pkg-config --cflags --libs opencv`

