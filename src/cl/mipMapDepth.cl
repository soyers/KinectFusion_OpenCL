#include "helpers.hpp"

__kernel void mipmapDepth_kernel(__global float* d_depthPyrsCurr, __global float* d_depthPyrsNext,__global float* d_gaussinKernel, float sigmaIcp, uint radius, uint w, uint h, float depthTrunc) //w and h are from depth next
{
    // calculate thread indices
    uint idxX = get_global_id(0);
    uint idxY = get_global_id(1);
    uint idx = idxX + w*idxY;

    // return if thread is out of bounds
    if(idxX >= w || idxY >= h) return;

    //Init buffer element
    d_depthPyrsNext[idxX + w*idxY] = 0.f;

    //Convolution center in the large image
    uint centerX = 2 * idxX;
    uint centerY = 2 * idxY;

    //Start points and end points in the large image
    uint xStart = centerX < radius ? 0 : centerX - radius;
    uint yStart = centerY < radius ? 0 : centerY - radius;
    uint xEnd = min(w*2, centerX + radius);
    uint yEnd = min(h*2, centerY + radius);

    //Apply convolution with gaussian kernel
    for(uint yIt = yStart; yIt < yEnd; ++yIt)
    {
        for(uint xIt = xStart; xIt < xEnd; ++xIt)
        {
            uint kernelIdx = (xIt - xStart) + (yIt - yStart) * (2*radius+1);
            if (isValidDepth(d_depthPyrsCurr[xIt + w*2*yIt], depthTrunc)) {
                    d_depthPyrsNext[idx] += d_gaussinKernel[kernelIdx] * d_depthPyrsCurr[xIt + w*2*yIt];
            }
        }
    }
}
