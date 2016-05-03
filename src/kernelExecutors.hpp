#ifndef KERNELEXECUTORS_HPP
#define KERNELEXECUTORS_HPP


#ifndef __OPENCL_C_VERSION__
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif
  #include <cmath>
#endif

#include "OpenclFrame.hpp"
#include "math.hpp"
#include "helpers.hpp"

void calculateVertexMap(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_depthMap, float3x3 kInv, cl_float depthTrunc, cl_uint w, cl_uint h);
void calculateNormalMap(OpenclFrame& clFrame, cl_mem& d_normalMap, cl_mem& d_vertexMap, cl_float derivativeTrunc, cl_uint w, cl_uint h);
void integrateTSDF(OpenclFrame& clFrame, cl_mem& d_values, cl_mem& d_weights, cl_mem& d_colors, cl_mem& d_depthMap, cl_mem& d_colorImage, float3x3 k, float4x4 transInv, cl_float maxTrunc, cl_float minTrunc, cl_float maxWeight, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize);
void raycastTSDF(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_normalMap, cl_mem& d_depthMap, cl_mem& d_colorMap, cl_mem& d_colorsTSDF, cl_mem& d_valuesTSDF, float3x3 k, float3x3 kInv, float4x4 trans, float4x4 transInv, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize);

#endif
