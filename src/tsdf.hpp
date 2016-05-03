#ifndef TSDF_HPP
#define TSDF_HPP

#include "OpenclFrame.hpp"
#include "clBuffer.hpp"
#include "kernelExecutors.hpp"
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif
#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

class tsdf
{    
    private:
        OpenclFrame& frame;
        clBuffer<cl_float> values;
        clBuffer<cl_float> weights;
        clBuffer<cl_float> colors;        
        cl_uint res;
        cl_float size;
    public:
        tsdf(OpenclFrame& clFrame, cl_uint res, cl_float size);
        void integrate(cl_mem& d_depthMap, cl_mem& d_colorImage, Eigen::Matrix3f k, Eigen::Matrix4f transInv, cl_float maxTrunc, cl_float minTrunc, cl_float maxWeight, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize);
        cl_uint volumeRes();
        cl_float volumeSize();
        clBuffer<cl_float>& valuesBuffer();
        clBuffer<cl_float>& colorsBuffer();
        ~tsdf();
};
#endif
