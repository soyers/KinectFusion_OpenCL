#ifndef ICP_HPP
#define ICP_HPP

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <cmath>
#include <opencv2/core/eigen.hpp>
#include "opencv2/rgbd/linemod.hpp"
#include <opencv2/rgbd.hpp>

#include "OpenclFrame.hpp"
#include "clBuffer.hpp"
#include "kernelExecutors.hpp"
#include "helpers.hpp"

class icp
{
    private:
        cv::Mat cameraMatrix;
        cv::Ptr<cv::rgbd::Odometry> cvICP;
        Eigen::Matrix4f currentPos;
        void generateGaussKernel();
    public:
        icp(cl_float depthTrunc, Eigen::Matrix3f k);
        void calculateTransform(cv::Mat currColor, cv::Mat currDepth, cv::Mat nextColor, cv::Mat nextDepth);
        Eigen::Matrix4f getCurrPos();
        ~icp();
};

#endif
