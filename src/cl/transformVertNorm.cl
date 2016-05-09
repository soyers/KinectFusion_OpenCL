#include "clMath.hpp"
#include "helpers.hpp"

__kernel void transformVertNorm_kernel(__global float* d_vertexMap, __global float* d_normalMap, float4x4 trans, float3x3 rot, uint w, uint h)
{
    uint idxX = get_global_id(0);
    uint idxY = get_global_id(1);

    // return if thread is out of bounds
    if (idxX >= w || idxY >= h) return;

    uint x = idxX + w * idxY;
    uint y = x + w*h;
    uint z = y + w*h;

    //Read input and transform it
    float3 untransformedVertex = (float3){d_vertexMap[x], d_vertexMap[y], d_vertexMap[z]};
    float3 untransformedNormal = (float3){d_normalMap[x], d_normalMap[y], d_normalMap[z]};
    float3 transformedVertex = transformVector(trans, untransformedVertex);
    float3 transformedNormal = mulMV3(rot, untransformedNormal);

    //Write transformed output
    d_vertexMap[x] = transformedVertex.x;
    d_vertexMap[y] = transformedVertex.y;
    d_vertexMap[z] = transformedVertex.z;
    d_normalMap[x] = transformedNormal.x;
    d_normalMap[y] = transformedNormal.y;
    d_normalMap[z] = transformedNormal.z;
}
