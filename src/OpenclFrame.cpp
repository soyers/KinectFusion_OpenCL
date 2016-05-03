#include "OpenclFrame.hpp"
#include "helpers.hpp"

cl_program createProgramFromSource(cl_context context, const std::string fileName, std::vector<cl_device_id> devices)
{	
	cl_int error;

    //Read source file into string
	std::ifstream ifs(fileName);
	std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>()); 

    // Prepare parameters for program creation   
	const char* sourceArray[1] = {programSource.data()};
    size_t lengthArray[1] = {programSource.size()};

    // Create program
	cl_program result = clCreateProgramWithSource (context, 1, sourceArray, lengthArray, &error);
	CHECK_OPENCL_CALL(error);

    // Build program
    error = clBuildProgram(result, 1, &devices[0], "-I src/", NULL, NULL);

    //Check build process
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(result, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(result, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("Error Log:\n%s\n", log);
        free(log);
        throw std::logic_error("Error");
    } else {
        CHECK_OPENCL_CALL(error);
    }

	return result;
}

OpenclFrame::OpenclFrame()
{
    //Disable compiler side source caching for NVidia cards since otherwise kernels are not recompiled if only an include of the kernel changes
    //See http://stackoverflow.com/questions/31338520/opencl-clbuildprogram-caches-source-and-does-not-recompile-if-included-source
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    initOpenclFrame();
}

void OpenclFrame::initOpenclFrame()
{
    //Query for availible platforms
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms (platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    //Stop if no platfrom was found
    if (platformCount == 0)
    {
        throw std::logic_error("No OpenCL platform found");
    }

    //Query for availible devices
    cl_uint deviceCount = 0;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
    std::vector<cl_device_id> devices(deviceCount);
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

    //Create context
    const cl_context_properties contextProps [] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platforms[0]), 0, 0};
    context = clCreateContext(contextProps, deviceCount, devices.data (), nullptr, nullptr, &error);
    CHECK_OPENCL_CALL(error);

    // Create OpenCL command queue
    queue = clCreateCommandQueue(context, devices[0], 0, &error);
    CHECK_OPENCL_CALL(error);

    // Build program files
    for (std::map<std::string, std::vector<std::string>>::iterator programIt = programKernelNameList.begin(); programIt != programKernelNameList.end(); ++programIt) {
        std::cout << "Creating program: " << programIt->first << std::endl;
        cl_program program = createProgramFromSource(context, programIt->first, devices);
        // Create kernels of each program
        for (std::vector<std::string>::iterator kernelIt = programIt->second.begin(); kernelIt != programIt->second.end(); ++kernelIt) {
            std::cout << "Compiling kernel: " << kernelIt->c_str() << std::endl;
            kernels[*kernelIt] = clCreateKernel(program, kernelIt->c_str(), &error);
            CHECK_OPENCL_CALL(error);
        }
        programs.push_back(program);
    }
}

OpenclFrame::~OpenclFrame()
  {
    // Finish queue
  	CHECK_OPENCL_CALL(clFlush(queue));
	CHECK_OPENCL_CALL(clFinish(queue));

    // Release all kernels
    for (std::map<std::string, cl_kernel>::iterator kernelIt = kernels.begin(); kernelIt != kernels.end(); ++kernelIt) {
        clReleaseKernel(kernelIt->second);
    }

    // Release all programs
    for (std::vector<cl_program>::iterator programIt = programs.begin(); programIt != programs.end(); ++programIt) {
        clReleaseProgram(*programIt);
    }

    // Release queue
    CHECK_OPENCL_CALL(clReleaseCommandQueue(queue));

    // Release context
    CHECK_OPENCL_CALL(clReleaseContext(context));
  }
