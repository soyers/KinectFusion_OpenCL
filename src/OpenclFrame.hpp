#ifndef OPENCL_FRAME_HPP
#define OPENCL_FRAME_HPP

#ifndef __OPENCL_C_VERSION__
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif
#endif

#include <string>
#include <vector>
#include <fstream>
#include <map>

#include <iostream>
#include <cstring>

class OpenclFrame
{
    public:
        OpenclFrame();
        ~OpenclFrame();
        std::map<std::string, cl_kernel> kernels;
        cl_command_queue queue;
        cl_context context;
    private:
        std::map<std::string, std::vector<std::string>> programKernelNameList
        {
            {"src/cl/vertexMap.cl", {"vertexMap_kernel"}},
            {"src/cl/normalMap.cl", {"normalMap_kernel"}},
            {"src/cl/tsdfIntegration.cl", {"tsdfIntegration_kernel"}},
            {"src/cl/raycastVolume.cl", {"raycastVolume_kernel"}}
        };        
        std::vector<cl_program> programs;
        cl_int error;
        void initOpenclFrame();
        
};

// Launch a kernel with
// CHECK_OPENCL_CALL(clEnqueueNDRangeKernel (queue, kernel, dimensions, offset, globalSize, localSize, 0, nullptr, nullptr));
#endif
