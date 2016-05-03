#include "raycaster.hpp"

raycaster::raycaster(OpenclFrame& clFrame, cl_uint w, cl_uint h, Eigen::Matrix3f k) : frame(clFrame), w(w), h(h), k(k), normalMap(frame, w*h*3), vertexMap(frame, w*h*3), colorMap(frame, w*h*3), depthMap(frame, w*h)
{
}

void raycaster::raycastVolume(tsdf& tsdfInstance, Eigen::Matrix4f trans)
{
    raycastTSDF(frame, vertexMap.deviceBuffer(), normalMap.deviceBuffer(), depthMap.deviceBuffer(), colorMap.deviceBuffer(), tsdfInstance.colorsBuffer().deviceBuffer(), tsdfInstance.valuesBuffer().deviceBuffer(), convertEigen(k), convertEigen(Eigen::Matrix3f(k.inverse())), convertEigen(trans), convertEigen(Eigen::Matrix4f(trans.inverse())), w, h, tsdfInstance.volumeRes(), tsdfInstance.volumeSize());
}

clBuffer<cl_float>& raycaster::normalMapBuffer()
{
    return normalMap;
}

clBuffer<cl_float>& raycaster::vertexMapBuffer()
{
    return vertexMap;
}

clBuffer<cl_float>& raycaster::colorMapBuffer()
{
    return colorMap;
}

clBuffer<cl_float>& raycaster::depthMapBuffer()
{
    return depthMap;
}

raycaster::~raycaster()
{
}
