#include "helpers.hpp"

__kernel void normalMap_kernel(__global float* d_normalMap, __global const float* d_vertexMap, float derivativeTrunc, uint w, uint h)
{
    // calculate thread indices
    uint idxX = get_global_id(0);
    uint idxY = get_global_id(1);

    // return if thread is out of bounds
    if (idxX >= w || idxY >= h) return;

    //calculate difference quotient
    uint idxXl = idxX > 0 ? idxX - 1 : 0;
    uint idxYl = idxY > 0 ? idxY - 1 : 0;
    uint idxXr = min(idxX + 1, w - 1);
    uint idxYr = min(idxY + 1, h - 1);

    //calculate indices
    uint x = idxX + w * idxY;
    uint y = x + w * h;
    uint z = y + w * h;

    uint xrx = idxXr + w * idxY;
    uint xry = xrx + w * h;
    uint xrz = xry + w * h;
    uint xlx = idxXl + w * idxY;
    uint xly = xlx + w * h;
    uint xlz = xly + w * h;
    uint yrx = idxX + w * idxYr;
    uint yry = yrx + w * h;
    uint yrz = yry + w * h;
    uint ylx = idxX + w * idxYl;
    uint yly = ylx + w * h;
    uint ylz = yly + w * h;

    //load values
    float3 xr = {d_vertexMap[xrx], d_vertexMap[xry], d_vertexMap[xrz]};
    float3 xl = {d_vertexMap[xlx], d_vertexMap[xly], d_vertexMap[xlz]};
    float3 yr = {d_vertexMap[yrx], d_vertexMap[yry], d_vertexMap[yrz]};
    float3 yl = {d_vertexMap[ylx], d_vertexMap[yly], d_vertexMap[ylz]};

    // Calculate central differences and truncate derivative in each direction
    float3 dx = (xr - xl) / 2.0f;
    float3 dy = (yr - yl) / 2.0f;
    dx.x = truncateDerivative(dx.x, derivativeTrunc);
    dx.y = truncateDerivative(dx.y, derivativeTrunc);
    dx.z = truncateDerivative(dx.z, derivativeTrunc);
    dy.x = truncateDerivative(dy.x, derivativeTrunc);
    dy.y = truncateDerivative(dy.y, derivativeTrunc);
    dy.z = truncateDerivative(dy.z, derivativeTrunc);

    float3 cr = normalize(cross(dx, dy));    
    d_normalMap[x] = cr.x;
    d_normalMap[y] = cr.y;
    d_normalMap[z] = cr.z;
}
