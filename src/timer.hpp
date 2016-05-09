#include <ctime>
#include "OpenclFrame.hpp"

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

class timer
{
    private:
        clock_t startTime, endTime;
        OpenclFrame& frame;
    public:
        timer(OpenclFrame& clFrame) : startTime(0), endTime(0), frame(clFrame) {}
        void start()
        {            
          	CHECK_OPENCL_CALL(clFlush(frame.queue));
	        CHECK_OPENCL_CALL(clFinish(frame.queue))
            startTime = clock();
        }

        void end()
        {            
          	CHECK_OPENCL_CALL(clFlush(frame.queue));
	        CHECK_OPENCL_CALL(clFinish(frame.queue))
            endTime = clock();
        }

        float measurement()
        {
            return float(endTime - startTime) / CLOCKS_PER_SEC;
        }
};
