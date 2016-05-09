#ifndef PARAMPROVIDER_HPP
#define PARAMPROVIDER_HPP

#include <boost/program_options.hpp>
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif
#include <exception>


namespace po = boost::program_options;

class helpCalled: public std::exception
{
    private:
        std::string message;
    public:
        helpCalled(std::string message) : message(message) {};
        virtual const char* what() const throw()
        {
            return message.c_str();
        }
};

class paramProvider
{
    public:
        paramProvider(const int argc, const char * const argv[]);
        std::string inputPath();
        std::string assocFilename();
        cl_uint volumeRes();
        cl_float volumeSize();
        cl_float depthTruncation();
        cl_float normalDerivTrunc();
        cl_float tsdfMaxTrunc();
        cl_float tsdfMinTrunc();
        bool isUseICP();
    private: 
        po::options_description desc;
        std::string m_inputPath;
        std::string m_assocFilename;
        cl_uint m_volumeRes;
        cl_float m_volumeSize;
        cl_float m_depthTruncation;
        cl_float m_normalDerivTrunc;
        cl_float m_tsdfMaxTrunc;
        cl_float m_tsdfMinTrunc;
        bool m_useICP;
};

#endif
