#include "tsdf.hpp"

tsdf::tsdf(OpenclFrame& clFrame, cl_uint res, cl_float size) : frame(clFrame), res(res), size(size), values(frame, res*res*res), weights(frame, res*res*res), colors(frame, res*res*res*3)
{
    values.fillHost(-1.f);
    values.upload();
    weights.memsetHost(0);
    weights.upload();
    colors.memsetHost(0);
    colors.upload();
}

void tsdf::integrate(cl_mem& d_depthMap, cl_mem& d_colorImage, Eigen::Matrix3f k, Eigen::Matrix4f transInv, cl_float maxTrunc, cl_float minTrunc, cl_float maxWeight, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize)
{
    integrateTSDF(frame, values.deviceBuffer(), weights.deviceBuffer(), colors.deviceBuffer(), d_depthMap, d_colorImage, convertEigen(k), convertEigen(transInv), maxTrunc, minTrunc, maxWeight, w, h, res, size);
}

cl_uint tsdf::volumeRes()
{
    return res;
}

cl_float tsdf::volumeSize()
{
    return size;
}

clBuffer<cl_float>& tsdf::valuesBuffer()
{
    return values;
}

clBuffer<cl_float>& tsdf::colorsBuffer()
{
    return colors;
}

tsdf::~tsdf()
{
    
}
