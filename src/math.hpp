#ifndef MATH_HPP
#define MATH_HPP

#include <iostream>
#include <stdexcept>
#include <iomanip>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>


struct float3x3
{
   cl_float3 m0; //Row 1
   cl_float3 m1; //Row 2
   cl_float3 m2; //Row 3
};

struct float4x4
{
   cl_float4 m0; //Row 1
   cl_float4 m1; //Row 2
   cl_float4 m2; //Row 3
   cl_float4 m3; //Row 4
};


inline cl_float3 convertEigen(const Eigen::Vector3f& vec)
{
   cl_float3 result = (cl_float3){vec.x(), vec.y(), vec.z()};
   return result;
}

inline Eigen::Vector3f convertEigen(const cl_float3& vec)
{
   Eigen::Vector3f result;
   result << vec.x, vec.y, vec.z;
   return result;
}

inline float3x3 convertEigen(const Eigen::Matrix3f& mat)
{
   float3x3 result;
   result.m0 = (cl_float3){mat(0, 0), mat(0, 1), mat(0, 2)};
   result.m1 = (cl_float3){mat(1, 0), mat(1, 1), mat(1, 2)};
   result.m2 = (cl_float3){mat(2, 0), mat(2, 1), mat(2, 2)};
   return result;
}

inline Eigen::Matrix3f convertEigen(const float3x3& mat)
{
    Eigen::Matrix3f result;
    result << mat.m0.x, mat.m0.y, mat.m0.z,
              mat.m1.x, mat.m1.y, mat.m1.z,
              mat.m2.x, mat.m2.y, mat.m2.z;
    return result;
}

inline float4x4 convertEigen(const Eigen::Matrix4f& value)
{
   float4x4 result;
   result.m0 = (cl_float3){value(0, 0), value(0, 1), value(0, 2), value(0, 3)};
   result.m1 = (cl_float3){value(1, 0), value(1, 1), value(1, 2), value(1, 3)};
   result.m2 = (cl_float3){value(2, 0), value(2, 1), value(2, 2), value(2, 3)};
   result.m3 = (cl_float3){value(3, 0), value(3, 1), value(3, 2), value(3, 3)};
   return result;
}

inline Eigen::Matrix4f convertEigen(const float4x4& mat)
{
    Eigen::Matrix4f result;
    result << mat.m0.x, mat.m0.y, mat.m0.z, mat.m0.w,
              mat.m1.x, mat.m1.y, mat.m1.z, mat.m1.w,
              mat.m2.x, mat.m2.y, mat.m2.z, mat.m2.w,
              mat.m3.x, mat.m3.y, mat.m3.z, mat.m3.w;
    return result;
}

// Do not define cl_float3 operators sice cl_float3 is the same as cl_float4 (data alignment)
#define _MAKE_BINARY_OP(type, op, cop) \
  inline type##2 operator op(const type##2& a, const type##2& b) { return ( type##2){a.x op b.x, a.y op b.y}; } \
  inline type##4 operator op(const type##4& a, const type##4& b) { return ( type##4 ){a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w}; } \
  inline type##2& operator cop(type##2& a, const type##2& b) { return a = a op b; } \
  inline type##4& operator cop(type##4& a, const type##4& b) { return a = a op b; } \
  inline type##2 operator op(const type##2& a, type b) { return (type##2){a.x op b, a.y op b}; } \
  inline type##4 operator op(const type##4& a, type b) { return (type##4){a.x op b, a.y op b, a.z op b, a.w op b}; } \
  inline type##2 operator op(type a, const type##2& b) { return b op a; } \
  inline type##4 operator op(type a, const type##4& b) { return b op a; } \
  inline type##2& operator cop(type##2& a, type b) { return a = a op b; } \
  inline type##4& operator cop(type##4& a, type b) { return a = a op b; }
  
_MAKE_BINARY_OP(cl_float, +, +=)
_MAKE_BINARY_OP(cl_float, -, -=)
_MAKE_BINARY_OP(cl_float, *, *=)
_MAKE_BINARY_OP(cl_float, /, /=)
_MAKE_BINARY_OP(cl_int, +, +=)
_MAKE_BINARY_OP(cl_int, -, -=)
_MAKE_BINARY_OP(cl_int, *, *=)
_MAKE_BINARY_OP(cl_int, /, /=)
  
#undef _MAKE_BINARY_OP

//Define min and max on non vector types
inline cl_int min(const cl_int& a, const cl_int& b) {return a < b ? a : b;}
inline cl_float min(const cl_float& a, const cl_float b) {return a < b ? a : b;}
inline cl_int max(const cl_int& a, const cl_int b) {return a > b ? a : b;}
inline cl_float max(const cl_float& a, const cl_float b) {return a > b ? a : b;}

#define _MAKE_MINMAX(type, name) \
  inline type##2 name(const type##2& a, const type##2& b) { return (type##2){name(a.x, b.x), name(a.y, b.y)}; } \
  inline type##4 name(const type##4& a, const type##4& b) { return (type##4){name(a.x, b.x), name(a.y, b.y), name(a.z, b.z), name(a.w, b.w)}; }

_MAKE_MINMAX(cl_float, min)
_MAKE_MINMAX(cl_float, max)
_MAKE_MINMAX(cl_int, min)
_MAKE_MINMAX(cl_int, max)
  
#undef _MAKE_MINMAX

#define _MAKE_UNARY_OP(type) \
  inline type##2 operator-(const type##2& x) { return (type##2){-x.x, -x.y}; } \
  inline type##4 operator-(const type##4& x) { return (type##4){-x.x, -x.y, -x.z, -x.w}; } \
  inline type##2 operator+(const type##2& x) { return x; } \
  inline type##4 operator+(const type##4& x) { return x; }

_MAKE_UNARY_OP(cl_float)
_MAKE_UNARY_OP(cl_int)

#undef _MAKE_UNARY_OP

inline float dot(const cl_float4 a, const cl_float4 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline cl_float3 operator*(const float3x3& mat, const cl_float3& vec)
{
  return (cl_float3){dot(mat.m0, vec), dot(mat.m1, vec), dot(mat.m2, vec)};
}

inline cl_float4 operator*(const float4x4& mat, const cl_float4& vec)
{
  return (cl_float4){dot(mat.m0, vec), dot(mat.m1, vec), dot(mat.m2, vec), dot(mat.m3, vec)};
}

inline float length(const cl_float3& vec)
{
    return sqrt(dot(vec, vec));
}

inline cl_float3 cross(const cl_float3& a, const cl_float3& b)
{
  return (cl_float3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline cl_float3 normalize(const cl_float3& vec)
{
    return vec / length(vec);
}

inline cl_float3 transformVector(const float4x4 m, const cl_float3 v)
{
  cl_float4 r = (cl_float4){v.x, v.y, v.z, 1};
  r = m * r;
  return (cl_float3){r.x, r.y, r.z};
}

inline cl_float3 transformNormal(const float4x4& m, const cl_float3& v)
{
  float3x3 t;
  t.m0 = (cl_float3){m.m0.x, m.m0.y, m.m0.z};
  t.m1 = (cl_float3){m.m1.x, m.m1.y, m.m1.z};
  t.m2 = (cl_float3){m.m2.x, m.m2.y, m.m2.z};
  return normalize(t * normalize(v));
}

namespace detail
{
    struct ios_state_saver
    {
    private:
      std::ostream& m_stream;
      std::ios::fmtflags m_flags;
      
      ios_state_saver(const ios_state_saver&);
      ios_state_saver& operator=(const ios_state_saver&);
      
    public:
      explicit ios_state_saver(std::ostream& s) :
	      m_stream(s),
	      m_flags(s.flags())
      {
      }
      
      ~ios_state_saver()
      {
	    this->m_stream.flags(this->m_flags);
      }
    };
  
    inline std::ostream& write(std::ostream& s, float v)
    {
      return s << std::setfill(' ') << std::setw(7) << std::setprecision(5) << std::setiosflags(std::ios::showpos) << std::fixed << v;
    }
    
    inline std::ostream& write(std::ostream& s, int32_t v)
    {
      return s << std::setfill(' ') << std::setw(sizeof(v) * 3) << std::dec << v;
    }
    
    inline std::ostream& write(std::ostream& s, const char* v)
    {
      return s << v;
    }
    
    template<typename T>
    inline std::ostream& write(std::ostream& s, T x, T y)
    {
      ios_state_saver state(s);
      return write(write(write(write(write(s, "["), x), ", "), y), "]");
    }
    
    template<typename T>
    inline std::ostream& write(std::ostream& s, T x, T y, T z)
    {
      ios_state_saver state(s);
      return write(write(write(write(write(write(write(s, "["), x), ", "), y), ", "), z), "]");
    }
    
    template<typename T>
    inline std::ostream& write(std::ostream& s, T x, T y, T z, T w)
    {
      ios_state_saver state(s);
      return write(write(write(write(write(write(write(write(write(s, "["), x), ", "), y), ", "), z), ", "), w), "]");
    }
}

inline std::ostream& operator<<(std::ostream& s, const cl_int2& v)
{
  return detail::write(s, v.x, v.y);
}

inline std::ostream& operator<<(std::ostream& s, const cl_int4& v)
{
  return detail::write(s, v.x, v.y, v.z, v.w);
}

inline std::ostream& operator<<(std::ostream& s, const cl_float2& v)
{
  return detail::write(s, v.x, v.y);
}

inline std::ostream& operator<<(std::ostream& s, const cl_float4& v)
{
  return detail::write(s, v.x, v.y, v.z, v.w);
}

inline std::ostream& operator<<(std::ostream& s, const float3x3& v)
{
  return s << "[" << v.m0 << ", " << v.m1 << ", " << v.m2 << "]";
}

inline std::ostream& operator<<(std::ostream& s, const float4x4& v)
{
  return s << "[" << v.m0 << ", " << v.m1 << ", " << v.m2 << ", " << v.m3 << "]";
}

#endif
