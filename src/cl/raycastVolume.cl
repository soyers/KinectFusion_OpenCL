#include "clMath.hpp"
#include "helpers.hpp"

#define RAYCASTER_DIR_MIN_VALUE       0.000001f
#define RAYCASTER_MAX_REFINEMENT      0

inline bool checkUpperBounds(float i, float3 rayStart, float3 rayDir, uint volumeRes)
{
    float3 tmp = rayStart + i * rayDir;
    return tmp.x <= volumeRes && tmp.y <= volumeRes && tmp.z <= volumeRes;
}

__kernel void raycastVolume_kernel(__global float* d_vertexMap, __global float* d_normalMap, __global float* d_depthMap, __global float* d_colorMap, __global const float* d_colorsTSDF, __global const float* d_valuesTSDF, float3x3 k, float3x3 kInv, float4x4 trans, float4x4 transInv, uint w, uint h, uint volumeRes, float volumeSize/*, uint refinementLevels*/)
{
    // calculate thread indices
    uint idxX = get_global_id(0);
    uint idxY = get_global_id(1);

    // return if thread is out of bounds
    if (idxX >= w || idxY >= h) return;

    // calculate output 3D indices
    uint x = idxX + w * idxY;
    uint y = x + w * h;
    uint z = y + w * h;

    // initialize vertexmap, depthmap and normalmap with NaN
    d_vertexMap[x] = NAN;
    d_vertexMap[y] = NAN;
    d_vertexMap[z] = NAN;
    d_normalMap[x] = NAN;
    d_normalMap[y] = NAN;
    d_normalMap[z] = NAN;
    d_depthMap[x] = NAN;
    // initialize colormap with 0.f
    d_colorMap[x] = 0.f;
    d_colorMap[y] = 0.f;
    d_colorMap[z] = 0.f;

    //Backproject, transform to global space and transform to voxel space (Input: (x,y,z) with x in [0..imgWidth], y in [0..imgHeight], z in [depth in meters]
    float3 rayStart = convertGlobalVoxel(transformVector(trans, mulMV3(kInv, (float3){0.f, 0.f, 0.f})), volumeRes, volumeSize);
    float3 rayNext = convertGlobalVoxel(transformVector(trans, mulMV3(kInv, (float3){(float)(idxX), (float)(idxY), 1.f})), volumeRes, volumeSize);
    float3 rayDir = normalize(rayNext - rayStart);

    float tsdfPrev = 0.f;
    float tsdfCurr = safeLoadTSDF(d_valuesTSDF, rayStart, volumeRes);
    float3 currPos;

    for(float i = 0.f; checkUpperBounds(i, rayStart, rayDir, volumeRes); i += 1.f) {
        currPos = rayStart + i * rayDir;
        tsdfPrev = tsdfCurr;
        tsdfCurr = safeLoadTSDF(d_valuesTSDF, currPos, volumeRes);
        if (tsdfPrev < 0 && tsdfCurr > 0) { //Zero crossing found
            //calculate vertex map
            float3 vertex = transformVector(transInv, convertVoxelGlobal(currPos, volumeRes, volumeSize));
            d_vertexMap[x] = vertex.x;
            d_vertexMap[y] = vertex.y;
            d_vertexMap[z] = vertex.z;

            //calculate depth
            //inverse of: vertex = (kInv * make_float3(x, y, 1.0f)) * depth;
            d_depthMap[x] = length(vertex) / length(mulMV3(kInv, (float3){idxX, idxY, 1.0f}));

            //Colors
            float3 color = safeLoadColor(d_colorsTSDF, currPos, volumeRes, volumeSize);
            d_colorMap[x] = color.x;
            d_colorMap[y] = color.y;
            d_colorMap[z] = color.z;

            //compute normal
            float3 posForward, normal, f1, f2;
            posForward = currPos;
            posForward.x -= 1.f;
            f1.x = safeLoadTSDF(d_valuesTSDF, posForward, volumeRes);
            posForward = currPos;
            posForward.y -= 1.f;
            f1.y = safeLoadTSDF(d_valuesTSDF, posForward, volumeRes);
            posForward = currPos;
            posForward.z -= 1.f;
            f1.z = safeLoadTSDF(d_valuesTSDF, posForward, volumeRes);
            posForward = currPos;
            posForward.x += 1.f;
            f2.x = safeLoadTSDF(d_valuesTSDF, posForward, volumeRes);
            posForward = currPos;
            posForward.y += 1.f;
            f2.y = safeLoadTSDF(d_valuesTSDF, posForward, volumeRes);
            posForward = currPos;
            posForward.z += 1.f;
            f2.z = safeLoadTSDF(d_valuesTSDF, posForward, volumeRes);
            if(isnan(f1.x) || isnan(f1.y) || isnan(f1.z) || isnan(f2.x) || isnan(f2.y) || isnan(f2.z)) return;
            normal = f2 - f1;
            normal = normalize(normal);

            //write normal to buffer
            float3 transformedNormal = transformNormal(transInv, normal);
            d_normalMap[x] = transformedNormal.x;
            d_normalMap[y] = transformedNormal.y;
            d_normalMap[z] = transformedNormal.z;
            break;
        }
    }
}
