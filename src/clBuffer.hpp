#ifndef CLBUFFER_HPP
#define CLBUFFER_HPP

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#include "OpenclFrame.hpp"
#include "helpers.hpp"

template<typename T>
struct clBuffer
{
    private:
        cl_mem d_buffer;
        T* h_buffer;
        size_t bufferSize;
        OpenclFrame& frame;

        //make non-copyable
        clBuffer(const clBuffer&);
        clBuffer& operator=(const clBuffer&);

    public:
        clBuffer(OpenclFrame& frame, size_t size) : frame(frame), bufferSize(size), h_buffer(new T[size])
        {
            d_buffer = clCreateBuffer(frame.context, CL_MEM_READ_WRITE, sizeof(T) * size, NULL, NULL);
        }

        clBuffer(OpenclFrame& frame, size_t size, T* data) : frame(frame), bufferSize(size), h_buffer(new T[size])
        {
            d_buffer = clCreateBuffer(frame.context, CL_MEM_READ_WRITE, sizeof(T) * size, NULL, NULL);
            memcpy(h_buffer, data, sizeof(T) * size);
            upload();
        }

        ~clBuffer()
        {
            delete[] h_buffer;
            CHECK_OPENCL_CALL(clReleaseMemObject(d_buffer));
        }

        void upload()
        {
            clEnqueueWriteBuffer(frame.queue, d_buffer, CL_TRUE, 0, sizeof(T) * size(), h_buffer, 0, NULL, NULL);
        }

        void download()
        {
            clEnqueueReadBuffer(frame.queue, d_buffer, CL_TRUE, 0, sizeof(T) * size(), h_buffer, 0, NULL, NULL);
        }

        size_t size()
        {
            return bufferSize;
        }

        void memsetHost(int32_t value)
        {
            memset(h_buffer, value, size() * sizeof(T));
        }

        void fillHost(T value)
        {
            std::fill_n(h_buffer, size(), value);
        }

        T* hostBuffer()
        {
            return h_buffer;
        }

        cl_mem& deviceBuffer()
        {
            return d_buffer;
        }
};
#endif
