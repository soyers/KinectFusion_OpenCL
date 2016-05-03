#ifndef RGBDBENCHTUM_HPP
#define RGBDBENCHTUM_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

class rgbdBenchTUM
{
    private:
        std::string inputFolder;
        std::vector<std::string> inputLines;
        cv::Mat currDepth;
        cv::Mat currColor;
        Eigen::Matrix4f currPos;
        cl_uint currentLineNumber;
    public:
        rgbdBenchTUM(const std::string inputFolder, const std::string& inputFilename);
        bool fetchData(cl_uint lineNumber);
        cv::Mat currentDepth();
        cv::Mat currentColor();
        Eigen::Matrix4f currentPos();
        cl_uint size();
        cl_uint currentLine();
        ~rgbdBenchTUM();
};
#endif
