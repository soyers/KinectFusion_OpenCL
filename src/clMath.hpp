#ifndef CLMATH_HPP
#define CLMATH_HPP

typedef struct
{
   float3 m0; //Row 1
   float3 m1; //Row 2
   float3 m2; //Row 3
} float3x3;

typedef struct
{
   float4 m0; //Row 1
   float4 m1; //Row 2
   float4 m2; //Row 3
   float4 m3; //Row 4
} float4x4;

//This function is necessary since opencl does not yet support operator overloading
float4 mulMV4(const float4x4 mat, const float4 vec)
{
    return (float4){dot(mat.m0, vec), dot(mat.m1, vec), dot(mat.m2, vec), dot(mat.m3, vec)};
}

//This function is necessary since opencl does not yet support operator overloading
float3 mulMV3(const float3x3 mat, const float3 vec)
{
    return (float3){dot(mat.m0, vec), dot(mat.m1, vec), dot(mat.m2, vec)};
}

float3 transformVector(const float4x4 m, const float3 v)
{
  float4 r = (float4){v.x, v.y, v.z, 1};
  r = mulMV4(m,r);
  return (float3){r.x, r.y, r.z};
}

float3 transformNormal(const float4x4 m, const float3 v)
{
  float3x3 t;
  t.m0 = (float3){m.m0.x, m.m0.y, m.m0.z};
  t.m1 = (float3){m.m1.x, m.m1.y, m.m1.z};
  t.m2 = (float3){m.m2.x, m.m2.y, m.m2.z};
  return normalize(mulMV3(t, normalize(v)));
}

#endif
