KinectFusion OpenCL implementation
==============

KinectFusion is an approach for reconstructing rigid geometric shapes using a consumer-grade RGB-D camera.
This repository is a development repository for creating an OpenCL implementation of the approach.

# Installation
## Prerequisites
Although most of the dependencies can be automatically downloaded using the [setup_external.sh](setup_external.sh) script, the project requires OpenCL to be installed correctly.

## Dependencies
The Project currently depends on the OopenCV library wnd its contributers's modules that can be automatically downloaded and installed using the [setup_external.sh](setup_external.sh) script in the project folder.
The following libraries are installed from the ubuntu software repository or downloaded, built from source and installed into the [libs](libs/) directory:
- [OpenCV](https://github.com/opencv/opencv) with extra modules from [OpenCV Contrib](https://github.com/opencv/opencv_contrib)

## Datasets
The project is able to read the assoociated rgb and depth files from all sequences in the [RGB-D SLAM Dataset and Benchmark](http://vision.in.tum.de/data/datasets/rgbd-dataset).
Furthermore it provides download scripts [downloadFreiburg.py](data/downloadFreiburg.py) for the [fr1/xyz](http://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz) dataset that also performs frame association.
The dataset is downloaded to the [data](data/) folder.

## Build and run
The project is simply built with the command `cmake .` in the project directory followed by `make`. Note that building in a separate build folder requres copying the [OpenCL kernels](src/cl) manually to `<build-folder>/src/cl`.
Executing the project with `./KinectFusion_OpenCL` will run the reconstruction on the downloaded dataset.

# References
- [KinectFusion Paper](https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwic1fPeorzXAhXQCuwKHZJ-D-AQFggnMAA&url=https%3A%2F%2Fwww.microsoft.com%2Fen-us%2Fresearch%2Fwp-content%2Fuploads%2F2016%2F02%2Fismar2011.pdf&usg=AOvVaw3uHY0TJIr3p57KW4p52rtC)
- [OpenCV](http://opencv.org)
