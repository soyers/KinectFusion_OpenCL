#ifndef RAYCASTER_HPP
#define RAYCASTER_HPP

#include "OpenclFrame.hpp"
#include "clBuffer.hpp"
#include "kernelExecutors.hpp"
#include "tsdf.hpp"
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif
#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

class raycaster
{    
    private:
        OpenclFrame& frame;
        cl_uint w,h;
        clBuffer<cl_float> normalMap;
        clBuffer<cl_float> vertexMap;        
        clBuffer<cl_float> colorMap;
        clBuffer<cl_float> depthMap;
        Eigen::Matrix3f k;
    public:
        raycaster(OpenclFrame& clFrame, cl_uint w, cl_uint h, Eigen::Matrix3f k);
        void raycastVolume(tsdf& tsdfInstance, Eigen::Matrix4f trans);
        clBuffer<cl_float>& normalMapBuffer();
        clBuffer<cl_float>& vertexMapBuffer();
        clBuffer<cl_float>& colorMapBuffer();
        clBuffer<cl_float>& depthMapBuffer();
        ~raycaster();
};
#endif
