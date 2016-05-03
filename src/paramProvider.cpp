#include "paramProvider.hpp"

paramProvider::paramProvider(const int argc, const char * const argv[]) : desc("Options")
{
    // Declare the supported options
    desc.add_options() 
        ("help,h",
            "Print help message") 
        //("input,i", po::value<std::string>(&m_inputPath)->required(),
        //    "Input directory")
        ("input,i", po::value<std::string>(&m_inputPath)->default_value(std::string(PROJECT_DIR) + "/data/rgbd_dataset_freiburg1_xyz/"),
            "Input directory")
        ("filename,f", po::value<std::string>(&m_assocFilename)->default_value(std::string(PROJECT_DIR) + "rgbd_assoc_poses.txt"),
            "Name of association file inside input directory")
        ("volume-res,r", po::value<cl_uint>(&m_volumeRes)->default_value(256),
            "Resolution of the voxel grid")
        ("volume-size,s", po::value<cl_float>(&m_volumeSize)->default_value(3.0f),
            "Size of the voxel grid in meters")
        ("depth-truncation,t", po::value<cl_float>(&m_depthTruncation)->default_value(3.0f),
            "Maximum considered distance in meters")
        ("normal-deriv-trunc,n", po::value<cl_float>(&m_normalDerivTrunc)->default_value(0.3f),
            "Derivative truncation value during normal map calculation in meters")
        ("tsdf-max-trunc,l", po::value<cl_float>(&m_tsdfMaxTrunc)->default_value(0.03f),
            "Upper truncation value for tsdf in meters")
        ("tsdf-min-trunc,u", po::value<cl_float>(&m_tsdfMinTrunc)->default_value(0.03f),
            "Lower truncation value for tsdf in meters");
    
    po::variables_map vm;
    try
    {
        //Try to populate the option variables
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        //If help option is set print help and exit
        if (vm.count("help"))
        {
            std::stringstream ss;
            ss << desc << std::endl;
            throw helpCalled(ss.str());
        }
    } 
    catch (const boost::program_options::required_option & e)
    {
        //If help is set and required options are missing, print help
        if (vm.count("help"))
        {
            std::stringstream ss;
            ss << desc << std::endl;
            throw helpCalled(ss.str());
        }
        else
        {
            // If requred parameters are missing and help is missing, print error
            throw e;
        }
    }

    //Print parsed parameters
    std::cout << "Parameters parsed: " <<  std::endl;
    std::cout << "    Filename: " << m_assocFilename << std::endl;
    std::cout << "    m_inputPath: " << m_inputPath << std::endl;
    std::cout << "    m_volumeRes: " << m_volumeRes << std::endl;
    std::cout << "    m_volumeSize: " << m_volumeSize << std::endl;
    std::cout << "    m_depthTruncation: " << m_depthTruncation << std::endl;
    std::cout << "    m_normalDerivTrunc: " << m_normalDerivTrunc << std::endl;
    std::cout << "    m_tsdfMaxTrunc: " << m_tsdfMaxTrunc << std::endl;
    std::cout << "    m_tsdfMinTrunc: " << m_tsdfMinTrunc << std::endl;
}

std::string paramProvider::assocFilename()
{
    return m_assocFilename;
}

std::string paramProvider::inputPath()
{
    return m_inputPath;
}

cl_uint paramProvider::volumeRes()
{
    return m_volumeRes;
}

cl_float paramProvider::volumeSize()
{
    return m_volumeSize;
}

cl_float paramProvider::depthTruncation()
{
    return m_depthTruncation;
}

cl_float paramProvider::normalDerivTrunc()
{
    return m_normalDerivTrunc;
}

cl_float paramProvider::tsdfMaxTrunc()
{
    return m_tsdfMaxTrunc;
}

cl_float paramProvider::tsdfMinTrunc()
{
    return m_tsdfMinTrunc;
}

