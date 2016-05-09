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

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include "OpenclFrame.hpp"
#include "math.hpp"
#include "helpers.hpp"

void calculateVertexMap(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_depthMap, float3x3 kInv, cl_float depthTrunc, cl_uint w, cl_uint h);
void calculateNormalMap(OpenclFrame& clFrame, cl_mem& d_normalMap, cl_mem& d_vertexMap, cl_float derivativeTrunc, cl_uint w, cl_uint h);
void integrateTSDF(OpenclFrame& clFrame, cl_mem& d_values, cl_mem& d_weights, cl_mem& d_colors, cl_mem& d_depthMap, cl_mem& d_colorImage, float3x3 k, float4x4 transInv, cl_float maxTrunc, cl_float minTrunc, cl_float maxWeight, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize);
void raycastTSDF(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_normalMap, cl_mem& d_depthMap, cl_mem& d_colorMap, cl_mem& d_colorsTSDF, cl_mem& d_valuesTSDF, float3x3 k, float3x3 kInv, float4x4 trans, float4x4 transInv, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize);
void mipMap(OpenclFrame& clFrame, cl_mem& d_depthPyrsCurr, cl_mem& d_depthPyrsNext, cl_mem& d_gaussinKernel, cl_float sigmaIcp, cl_uint radius, cl_uint w, cl_uint h, cl_float depthTrunc);
void transformVertNorm(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_normalMap, Eigen::Matrix4f trans, cl_uint w, cl_uint h);
void findCorrespondences(OpenclFrame& clFrame, cl_mem& d_vertexMapCurr, cl_mem& d_normalMapCurr, cl_mem& d_vertexMapNext, cl_mem& d_normalMapNext, cl_mem& d_correspondences, Eigen::Matrix4f currPosInv, Eigen::Matrix4f nextPos, Eigen::Matrix3f kI, cl_float distanceThreshold, cl_float normalThreshold, cl_uint w, cl_uint h);

#endif
