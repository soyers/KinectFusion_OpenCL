#ifndef HELPERS_HPP
#define HELPERS_HPP

#ifndef __OPENCL_C_VERSION__
    #include <iostream>
    #include <stdexcept>
    #include <opencv2/highgui/highgui.hpp>
    #include "math.hpp"
    #ifdef __APPLE__
        #include "OpenCL/opencl.h"
    #else
        #include "CL/cl.h"
    #endif
        #include <cmath>
#else
    #include "clMath.hpp"
#endif

//Make fuctions containing vector types usable on host and device
#ifdef __OPENCL_C_VERSION__
typedef float cl_float;
typedef float2 cl_float2;
typedef float3 cl_float3;
typedef float4 cl_float4;
typedef int cl_int;
typedef int2 cl_int2;
typedef int3 cl_int3;
typedef int4 cl_int4;
typedef uint cl_uint;
#endif

//helper call that checks for opencl-errors
#define CHECK_OPENCL_CALL(error) { \
	if (error != CL_SUCCESS) { \
        std::cout << __FILE__ << ", line " << __LINE__ << ": " << error << std::endl; \
        throw std::logic_error("Error"); \
	} }



#ifndef __OPENCL_C_VERSION__
//################################# Host Only functions #################################

inline float sign(float v)
{
    if (v > 0) return 1.f;
    else return 1.f;
}

// OpenCV image handling
inline void layer(cl_float* dst, const float* src, cl_uint w, cl_uint h, cl_uint nc)
{
    for (cl_uint c = 0; c < nc; ++c)
    {
        cl_uint cOffset = w*h*c;
        for (cl_uint x = 0; x < w; ++x)
        {
            for (cl_uint y = 0; y < h; ++y)
            {
                cl_uint posFlattened = x + w*y;
                dst[posFlattened + cOffset] = (cl_float)src[nc * posFlattened + nc - 1 - c];
            }
        }
    }
}

inline void layer(cl_float* dst, const cv::Mat& src)
{
    layer(dst, (float*)src.data, src.cols, src.rows, src.channels());
}

inline void interleave(float* dst, const cl_float* src, cl_uint w, cl_uint h, cl_uint nc)
{
    for (cl_uint c = 0; c < nc; ++c)
    {
        cl_uint cOffset = w*h*c;
        for (cl_uint x = 0; x < w; ++x)
        {
            for (cl_uint y = 0; y < h; ++y)
            {
                cl_uint posFlattened = x + w*y;
                dst[nc * posFlattened + nc - 1 - c] = static_cast<float>(src[posFlattened + cOffset]);
            }
        }
    }
}

inline void interleave(cv::Mat& dst, const cl_float *src)
{
    interleave((float*)dst.data, src, dst.cols, dst.rows, dst.channels());
}

inline void displayImage(const cv::Mat &image, const std::string winname, cl_uint x, cl_uint y)
{
    cv::namedWindow(winname, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(winname.c_str(), x, y);
    cv::imshow(winname, image);
}

inline void calcMinMax(const cl_float* data, cl_uint w, cl_uint h, cl_uint nc, cl_float& minVal, cl_float& maxVal)
{
    maxVal = NAN;
    minVal = NAN;
    for (cl_uint i = 0; i < w* h * nc; ++i)
    {
        if(!isnan(data[i])) maxVal = isnan(maxVal) ? data[i] : max(maxVal, data[i]);
        if(!isnan(data[i])) minVal = isnan(minVal) ? data[i] : min(minVal, data[i]);
    }
}

inline void show3DData(const cl_float* data, cl_uint w, cl_uint h, const char* winname, cl_uint x, cl_uint y)
{      
  cl_float maxValue;
  cl_float minValue;
  calcMinMax(data, w, h, 3, minValue, maxValue);
  cv::Mat mData(h, w, CV_32FC3);
  interleave(mData, data);
  mData = (mData - minValue) / (maxValue - minValue);
  displayImage(mData, winname, x, y);
}

inline cl_float3 calculateCentroid(const cl_float* vertexMap, cl_uint w, cl_uint h) 
{
    cl_float3 result = (cl_float3){0.f, 0.f, 0.f};
    cl_uint counter = 0;
    for (cl_uint i = 0; i < w; ++i) {
        for (cl_uint j = 0; j < h; ++j) {
            cl_float3 currVertex = (cl_float3){vertexMap[i + w*j], vertexMap[i + w*j + w*h], vertexMap[i + w*j + w*h*2]};
            if (!isnan(currVertex.x) && !isnan(currVertex.y) && !isnan(currVertex.z)) {
                result += currVertex;
                ++counter;
            }
        }
    }
    result /= counter;
    return result;
}

#endif

//################################# Host AND Device functions #################################

inline float truncateDerivative(cl_float dev, cl_float threshold)
{
    return threshold < fabs(dev) ? threshold * sign(dev) : dev;
}

inline bool isValidDepth(cl_float v, cl_float depthTrunc)
{
    return !(isnan(v) || v <= 0.01f || v > depthTrunc);
}

inline cl_float3 convertVoxelGlobal(cl_float3 point, cl_uint volumeRes, cl_float volumeSize) 
{
    // Calculate size of one voxel
    float voxelSize = volumeSize / volumeRes;
    return (point * voxelSize) - volumeSize / 2;
}

inline cl_float3 convertGlobalVoxel(cl_float3 point, cl_uint volumeRes, cl_float volumeSize) 
{
    // Calculate size of one voxel
    float voxelSize = volumeSize / volumeRes;
    return (point + volumeSize / 2) / voxelSize;
}

inline bool getDepthAndColor(cl_float2 p, const cl_float* d_depthMap, const cl_float* d_colorMap, cl_uint w, cl_uint h, cl_float* depth, cl_float3* color)
{
    // use depth value of nearest neighbour
    cl_int px = (int)floor(p.x + 0.5f);
    cl_int py = (int)floor(p.y + 0.5f);
    if(px < 0 || py < 0 || px + 1 >= w || py + 1 >= h) return false;
    *depth = d_depthMap[px + py * w];
    if(isnan(*depth)) return false;
    if(*depth <= 0.f) return false;
    cl_uint nPixels = w*h;
    // Using this access pattern since "->" is not accepted by the AMD compiler
    (*color).x = d_colorMap[px + py * w];
    (*color).y = d_colorMap[px + py * w + nPixels];
    (*color).z = d_colorMap[px + py * w + nPixels * 2];
    return true;
}

inline cl_float2 projectPoint(cl_float3 vert, float3x3 k) 
{
    // Project to pixel space
#ifdef __OPENCL_C_VERSION__
    cl_float3 tmp = mulMV3(k, vert);
#else
    cl_float3 tmp = k * vert;
#endif 
    return (cl_float2){tmp.x / tmp.z, tmp.y / tmp.z};
}

inline cl_int3 getVoxelCoords(const cl_float3 pos, const cl_uint volumeRes, const cl_float volumeSize)
{
    float voxelSize = volumeSize / volumeRes;
    cl_float3 p = pos + volumeSize / 2;
    return (cl_int3){(cl_int)(floor(p.x / voxelSize)), (cl_int)((p.y / voxelSize)), (cl_int)((p.z / voxelSize))};
}

//################################# Device Only functions #################################

#ifdef __OPENCL_C_VERSION__
inline float loadTSDFValue(__global const float* d_valuesTSDF, const uint3 pos, const uint volumeRes)
{
  return d_valuesTSDF[pos.x + volumeRes * (pos.y + volumeRes * pos.z)];
}

inline float3 loadTSDFColor(__global const float* d_colorTSDF, const uint3 pos, const uint volumeRes)
{
  uint idxX = pos.x + volumeRes * (pos.y + volumeRes * pos.z);
  uint idxY = idxX + volumeRes * volumeRes * volumeRes;
  uint idxZ = idxY + volumeRes * volumeRes * volumeRes;
  return (float3){d_colorTSDF[idxX], d_colorTSDF[idxY], d_colorTSDF[idxZ]};
}

inline float interpolateTrilinearf(const float c000, const float c100, const float c010, const float c110, const float c001, const float c101, const float c011, const float c111, const float3 point)
{
    return 
      c000 * (1 - point.x) * (1 - point.y) * (1 - point.z) + 
      c100 * point.x * (1 - point.y) * (1 - point.z) + 
      c010 * (1 - point.x) * point.y * (1 - point.z) + 
      c110 * point.x * point.y * (1 - point.z) + 
      c001 * (1 - point.x) * (1 - point.y) * point.z + 
      c101 * point.x * (1 - point.y) * point.z + 
      c011 * (1 - point.x) * point.y * point.z + 
      c111 * point.x * point.y * point.z;
}

inline float3 interpolateTrilinearf3(const float3 c000, const float3 c100, const float3 c010, const float3 c110, const float3 c001, const float3 c101, const float3 c011, const float3 c111, const float3 point)
{
    return 
      c000 * (1 - point.x) * (1 - point.y) * (1 - point.z) + 
      c100 * point.x * (1 - point.y) * (1 - point.z) + 
      c010 * (1 - point.x) * point.y * (1 - point.z) + 
      c110 * point.x * point.y * (1 - point.z) + 
      c001 * (1 - point.x) * (1 - point.y) * point.z + 
      c101 * point.x * (1 - point.y) * point.z + 
      c011 * (1 - point.x) * point.y * point.z + 
      c111 * point.x * point.y * point.z;
}

inline float trilinearTSDFValues(__global const float* d_valuesTSDF, float3 point, const cl_uint volumeRes)
{
    uint3 floored = (uint3){(uint)point.x, (uint)point.y, (uint)point.z};
	uint3 ceiled = (uint3){(uint)ceil(point.x), (uint)ceil(point.y), (uint)ceil(point.z)};

    float c000 = loadTSDFValue(d_valuesTSDF, floored, volumeRes);
    float c100 = loadTSDFValue(d_valuesTSDF, (uint3){ceiled.x, floored.y, floored.z}, volumeRes);
    float c010 = loadTSDFValue(d_valuesTSDF, (uint3){floored.x, ceiled.y, floored.z}, volumeRes);
    float c110 = loadTSDFValue(d_valuesTSDF, (uint3){ceiled.x, ceiled.y, floored.z}, volumeRes);
    float c001 = loadTSDFValue(d_valuesTSDF, (uint3){floored.x, floored.y, ceiled.z}, volumeRes);
    float c101 = loadTSDFValue(d_valuesTSDF, (uint3){ceiled.x, floored.y, ceiled.z}, volumeRes);
    float c011 = loadTSDFValue(d_valuesTSDF, (uint3){floored.x, ceiled.y, ceiled.z}, volumeRes);
    float c111 = loadTSDFValue(d_valuesTSDF, (uint3){ceiled.x, ceiled.y, ceiled.z}, volumeRes);

    //Calculate position inside containing voxel [(point.x - floored.x) / (ceiled.x - floored.x), (point.y - floored.y) / (ceiled.y - floored.y), (point.z - floored.z) / (ceiled.z - floored.z)]
    //The denominator can be left out since it is 1 if ceiled.k != floored.k. If ceiled.k == floored.k then the numerator is 0 anyway and posInside.k should also be 0 then
    float3 posInside = point - (float3){(float)floored.x, (float)floored.y, (float)floored.z};

    return interpolateTrilinearf(c000, c100, c010, c110, c001, c101, c011, c111, posInside);

}

inline float3 trilinearTSDFColors(__global const float* d_colorTSDF, float3 point, const cl_uint volumeRes)
{
    uint3 floored = (uint3){(uint)point.x, (uint)point.y, (uint)point.z};
	uint3 ceiled = (uint3){(uint)ceil(point.x), (uint)ceil(point.y), (uint)ceil(point.z)};

    float3 c000 = loadTSDFColor(d_colorTSDF, floored, volumeRes);
    float3 c100 = loadTSDFColor(d_colorTSDF, (uint3){ceiled.x, floored.y, floored.z}, volumeRes);
    float3 c010 = loadTSDFColor(d_colorTSDF, (uint3){floored.x, ceiled.y, floored.z}, volumeRes);
    float3 c110 = loadTSDFColor(d_colorTSDF, (uint3){ceiled.x, ceiled.y, floored.z}, volumeRes);
    float3 c001 = loadTSDFColor(d_colorTSDF, (uint3){floored.x, floored.y, ceiled.z}, volumeRes);
    float3 c101 = loadTSDFColor(d_colorTSDF, (uint3){ceiled.x, floored.y, ceiled.z}, volumeRes);
    float3 c011 = loadTSDFColor(d_colorTSDF, (uint3){floored.x, ceiled.y, ceiled.z}, volumeRes);
    float3 c111 = loadTSDFColor(d_colorTSDF, (uint3){ceiled.x, ceiled.y, ceiled.z}, volumeRes);

    //Calculate position inside containing voxel [(point.x - floored.x) / (ceiled.x - floored.x), (point.y - floored.y) / (ceiled.y - floored.y), (point.z - floored.z) / (ceiled.z - floored.z)]
    //The denominator can be left out since it is 1 if ceiled.k != floored.k. If ceiled.k == floored.k then the numerator is 0 anyway and posInside.k should also be 0 then
    float3 posInside = point - (float3){(float)floored.x, (float)floored.y, (float)floored.z};

    return interpolateTrilinearf3(c000, c100, c010, c110, c001, c101, c011, c111, posInside);

}

inline float3 safeLoadColor(__global const float* d_colorTSDF, const cl_float3 pos, const cl_uint volumeRes, const cl_float volumeSize)
{
    //return zero vector if position is out of volume bounds
    if(pos.x <= 0 || pos.x + 1 >= volumeRes || pos.y <= 0 || pos.y + 1 >= volumeRes || pos.z <= 0 || pos.z + 1 >= volumeRes) return (float3){0, 0, 0};
    return trilinearTSDFColors(d_colorTSDF, pos, volumeRes);
}

inline float safeLoadTSDF(__global const float* d_valuesTSDF, const float3 pos, const cl_uint volumeRes)
{
    //return NaN if out of or on bounds
    if(pos.x <= 0 || pos.x + 1 >= volumeRes || pos.y <= 0 || pos.y + 1 >= volumeRes || pos.z <= 0 || pos.z + 1 >= volumeRes) return NAN;
    return trilinearTSDFValues(d_valuesTSDF, pos, volumeRes);
}
#endif
#endif
