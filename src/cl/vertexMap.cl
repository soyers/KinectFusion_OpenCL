#include "clMath.hpp"
#include "helpers.hpp"

__kernel void vertexMap_kernel(__global float* d_vertexMap, __global float* d_depthMap, float3x3 kInv, float depthTrunc, uint w, uint h)
{
    // calculate thread indices
    uint idxX = get_global_id(0);
    uint idxY = get_global_id(1);

    // return if thread is out of bounds
    if (idxX >= w || idxY >= h) return;

    uint x = idxX + w * idxY;
    uint y = x + w * h;
    uint z = y + w * h;

    // remove invalid depth values and set nan instead
    float3 vertex;
    if (!isValidDepth(d_depthMap[x], depthTrunc)) {
      d_depthMap[x] = NAN;
      vertex = (float3){NAN, NAN, NAN};
    }
    else {
      // calculate formula v_i(x,y) = D_i(x,y) * K^-1 * [x,y|1]
      float3 screen = mulMV3(kInv, (float3){idxX, idxY, 1.0f});
      vertex = screen * d_depthMap[x];
    }
    d_vertexMap[x] = vertex.x;
    d_vertexMap[y] = vertex.y;
    d_vertexMap[z] = vertex.z;
}
