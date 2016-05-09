#include "kernelExecutors.hpp"

#define WARP_SIZE 64

void calculateVertexMap(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_depthMap, float3x3 kInv, cl_float depthTrunc, cl_uint w, cl_uint h)
{
    cl_kernel kernel = clFrame.kernels["vertexMap_kernel"];
    cl_command_queue queue = clFrame.queue;

    //Set kernel arguments
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vertexMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_depthMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 2, sizeof(float3x3), &kInv));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 3, sizeof(cl_float), &depthTrunc));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 4, sizeof(cl_uint), &w));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 5, sizeof(cl_uint), &h));

    // Add kernel to execution queue
    size_t globalSize[3] = {((w + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE , 
                            ((h + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE, 
                            1}; //Pad global size to multiples of warpsize (AMD: 64, NVidia: 32) (requires boundary checks inside kernel)
    CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL)); 
}

void calculateNormalMap(OpenclFrame& clFrame, cl_mem& d_normalMap, cl_mem& d_vertexMap, cl_float derivativeTrunc, cl_uint w, cl_uint h)
{
    cl_kernel kernel = clFrame.kernels["normalMap_kernel"];
    cl_command_queue queue = clFrame.queue;

    //Set kernel arguments
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_normalMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_vertexMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 2, sizeof(cl_float), &derivativeTrunc));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 3, sizeof(cl_uint), &w));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 4, sizeof(cl_uint), &h));

    // Add kernel to execution queue
    size_t globalSize[3] = {((w + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE , 
                            ((h + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE, 
                            1}; //Pad global size to multiples of warpsize (AMD: 64, NVidia: 32) (requires boundary checks inside kernel)
    CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL)); 
}

void integrateTSDF(OpenclFrame& clFrame, cl_mem& d_values, cl_mem& d_weights, cl_mem& d_colors, cl_mem& d_depthMap, cl_mem& d_colorImage, float3x3 k, float4x4 transInv, cl_float maxTrunc, cl_float minTrunc, cl_float maxWeight, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize)
{
    cl_kernel kernel = clFrame.kernels["tsdfIntegration_kernel"];
    cl_command_queue queue = clFrame.queue;

    //Set kernel arguments
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_values));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_weights));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_colors));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_depthMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_colorImage));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 5, sizeof(float3x3), &k));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 6, sizeof(float4x4), &transInv));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 7, sizeof(cl_float), &maxTrunc));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 8, sizeof(cl_float), &minTrunc));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 9, sizeof(cl_float), &maxWeight));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 10, sizeof(cl_uint), &w));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 11, sizeof(cl_uint), &h));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 12, sizeof(cl_uint), &volumeRes));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 13, sizeof(cl_float), &volumeSize));

    // Add kernel to execution queue
    size_t globalSize[3] = {((volumeRes + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE , 
                            ((volumeRes + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE, 
                            ((volumeRes + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE}; //Pad global size to multiples of warpsize (AMD: 64, NVidia: 32) (requires boundary checks inside kernel)
    CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalSize, NULL, 0, NULL, NULL)); 
}

void raycastTSDF(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_normalMap, cl_mem& d_depthMap, cl_mem& d_colorMap, cl_mem& d_colorsTSDF, cl_mem& d_valuesTSDF, float3x3 k, float3x3 kInv, float4x4 trans, float4x4 transInv, cl_uint w, cl_uint h, cl_uint volumeRes, cl_float volumeSize)
{
    cl_kernel kernel = clFrame.kernels["raycastVolume_kernel"];
    cl_command_queue queue = clFrame.queue;

    //Set kernel arguments
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vertexMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_normalMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_depthMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_colorMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_colorsTSDF));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_valuesTSDF));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 6, sizeof(float3x3), &k));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 7, sizeof(float3x3), &kInv));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 8, sizeof(float4x4), &trans));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 9, sizeof(float4x4), &transInv));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 10, sizeof(cl_uint), &w));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 11, sizeof(cl_uint), &h));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 12, sizeof(cl_uint), &volumeRes));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 13, sizeof(cl_float), &volumeSize));

    // Add kernel to execution queue
    size_t globalSize[3] = {((w + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE , 
                            ((h + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE, 
                            1}; //Pad global size to multiples of warpsize (AMD: 64, NVidia: 32) (requires boundary checks inside kernel)
    CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL)); 
}

void mipMap(OpenclFrame& clFrame, cl_mem& d_depthPyrsCurr, cl_mem& d_depthPyrsNext, cl_mem& d_gaussinKernel, cl_float sigmaIcp, cl_uint radius, cl_uint w, cl_uint h, cl_float depthTrunc)
{
    cl_kernel kernel = clFrame.kernels["mipmapDepth_kernel"];
    cl_command_queue queue = clFrame.queue;

    //Set kernel arguments
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_depthPyrsCurr));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_depthPyrsNext));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_gaussinKernel));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 3, sizeof(cl_float), &sigmaIcp));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 4, sizeof(cl_uint), &radius));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 5, sizeof(cl_uint), &w));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 6, sizeof(cl_uint), &h));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 7, sizeof(cl_float), &depthTrunc));

    // Add kernel to execution queue
    size_t globalSize[3] = {((w + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE , 
                            ((h + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE, 
                            1}; //Pad global size to multiples of warpsize (AMD: 64, NVidia: 32) (requires boundary checks inside kernel)
    CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL)); 
}

void transformVertNorm(OpenclFrame& clFrame, cl_mem& d_vertexMap, cl_mem& d_normalMap, Eigen::Matrix4f trans, cl_uint w, cl_uint h)
{
    cl_kernel kernel = clFrame.kernels["transformVertNorm_kernel"];
    cl_command_queue queue = clFrame.queue;

    //Prepare rotation matrix
    float3x3 rot = convertEigen(Eigen::Matrix3f(trans.topLeftCorner<3, 3>()));
    float4x4 transform = convertEigen(trans);

    //Set kernel arguments
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vertexMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_normalMap));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 2, sizeof(float4x4), &transform));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 3, sizeof(float3x3), &rot));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 4, sizeof(cl_uint), &w));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 5, sizeof(cl_uint), &h));

    // Add kernel to execution queue
    size_t globalSize[3] = {((w + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE , 
                            ((h + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE, 
                            1}; //Pad global size to multiples of warpsize (AMD: 64, NVidia: 32) (requires boundary checks inside kernel)
    CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL)); 
}

void findCorrespondences(OpenclFrame& clFrame, cl_mem& d_vertexMapCurr, cl_mem& d_normalMapCurr, cl_mem& d_vertexMapNext, cl_mem& d_normalMapNext, cl_mem& d_correspondences, Eigen::Matrix4f currPosInv, Eigen::Matrix4f nextPos, Eigen::Matrix3f kI, cl_float distanceThreshold, cl_float normalThreshold, cl_uint w, cl_uint h)
{
    cl_kernel kernel = clFrame.kernels["findCorrespondences_kernel"];
    cl_command_queue queue = clFrame.queue;

    //Prepare rotation next matrix
    float3x3 k = convertEigen(kI);
    float3x3 rotNext = convertEigen(Eigen::Matrix3f(nextPos.topLeftCorner<3, 3>()));
    float4x4 transformNext = convertEigen(nextPos);
    float4x4 transformCurrInv = convertEigen(currPosInv);

    //Set kernel arguments
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vertexMapCurr));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_normalMapCurr));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_vertexMapNext));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_normalMapNext));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_correspondences));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 5, sizeof(float4x4), &transformCurrInv));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 6, sizeof(float4x4), &transformNext));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 7, sizeof(float3x3), &rotNext));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 8, sizeof(float3x3), &k));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 9, sizeof(cl_float), &distanceThreshold));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 10, sizeof(cl_float), &normalThreshold));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 11, sizeof(cl_uint), &w));
    CHECK_OPENCL_CALL(clSetKernelArg(kernel, 12, sizeof(cl_uint), &h));

    // Add kernel to execution queue
    size_t globalSize[3] = {((w + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE , 
                            ((h + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE, 
                            1}; //Pad global size to multiples of warpsize (AMD: 64, NVidia: 32) (requires boundary checks inside kernel)
    CHECK_OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL)); 
}
