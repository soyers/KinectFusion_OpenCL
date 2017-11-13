#!/bin/sh

# folder for externals
EXT="${PWD}/libs"
EXT_OPENCV="${EXT}/opencv"
EXT_SRCS="${EXT}/sources"
EXT_OPENCV_CONTRIB_MODULES="${EXT_SRCS}/opencv_contrib/modules"
echo "External folder: $EXT"
echo "External folder sources: $EXT_SRCS"
echo "External folder OpenCV: $EXT_OPENCV"
echo "External folder OpenCV_Contrib: $EXT_OPENCV_CONTRIB_MODULES"
if [ ! -d "$EXT" ]; then
        mkdir $EXT
fi
if [ ! -d "$EXT_SRCS" ]; then
        mkdir $EXT_SRCS
fi
cd $EXT_SRCS

#Install OpenCV
cd $EXT_SRCS
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build_opencv 
cd build_opencv
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=OFF -DCMAKE_INSTALL_PREFIX=$EXT_OPENCV -DOPENCV_EXTRA_MODULES_PATH=${EXT_OPENCV_CONTRIB_MODULES} -DBUILD_opencv_dnn_modern=OFF ..
make -j4
make install

cd $EXT
rm -rf $EXT_SRCS
