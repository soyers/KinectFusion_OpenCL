#include "clMath.hpp"
#include "helpers.hpp"

__kernel void tsdfIntegration_kernel(__global float* d_values, __global float* d_weights, __global float* d_colors, __global const float* d_depthMap, __global const float* d_colorImage, float3x3 k, float4x4 transInv, float maxTrunc, float minTrunc, float maxWeight, uint w, uint h, uint volumeRes, float volumeSize)
{
    // calculate thread indices
    uint idxX = get_global_id(0);
    uint idxY = get_global_id(1);
    uint idxZ = get_global_id(2);

    // return if thread is out of bounds
    if (idxX >= volumeRes || idxY >= volumeRes || idxZ >= volumeRes) return;

    // calculate the index of the current voxel
    uint ind = idxX + volumeRes * (idxY + volumeRes * idxZ);

    // compute global coordinates
    float3 vGlobalCoord = convertVoxelGlobal((float3){idxX, idxY, idxZ}, volumeRes, volumeSize);
    float3 vertex = transformVector(transInv, vGlobalCoord);

    // perspective project vertex coordinates
    float2 vProjected = projectPoint(vertex, k);

    // interpolate depth value with nearest neighbour
    int2 p = (int2){(int)floor(vProjected.x + 0.5f), (int)floor(vProjected.y + 0.5f)};

    // Stop if projected point is outside of camera frustum
    if (p.x < 0 || p.y < 0 || p.x + 1 >= w || p.y + 1 >= h || isnan(d_depthMap[p.x + p.y * w]) || vertex.z < 0.01f) return;

    //Calculate sdf and weight
    float sdf = vertex.z - d_depthMap[p.x + p.y * w];
    float weightNew;

    //truncate sdf
    if (sdf <= maxTrunc) {
        float tsdfNew;
        if (sdf > 0) {
      	    weightNew = 1.0f - sdf / maxTrunc;
            tsdfNew = min(1.0f, sdf / maxTrunc);
        } else {
      	    weightNew = 1.0f;
            tsdfNew = max(-1.0f, sdf / minTrunc);
        }

        //compute updated TSDF value
        d_values[ind] = (d_values[ind] * d_weights[ind] + tsdfNew * weightNew) / (d_weights[ind] + weightNew);

        //updade colors
        uint volumeNPixels = volumeRes * volumeRes * volumeRes;
        float3 oldColor = (float3){d_colors[ind], d_colors[ind + volumeNPixels], d_colors[ind + volumeNPixels * 2]};
        float3 color = (float3){d_colorImage[p.x + p.y * w], d_colorImage[p.x + p.y * w + w*h], d_colorImage[p.x + p.y * w + w*h*2]};
        color = (oldColor * d_weights[ind] + color * weightNew) / (d_weights[ind] + weightNew);
        d_colors[ind] = color.x;
        d_colors[ind + volumeNPixels] = color.y;
        d_colors[ind + volumeNPixels * 2] = color.z;

        //store the new weight
        d_weights[ind] += weightNew;
    }
}
