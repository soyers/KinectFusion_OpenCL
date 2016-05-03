#include "rgbdBenchTUM.hpp"

rgbdBenchTUM::rgbdBenchTUM(const std::string inputFolder, const std::string& inputFilename) : inputFolder(inputFolder)
{
    std::ifstream inStream(inputFolder + inputFilename);
    if (!inStream.is_open()) throw std::logic_error("Unable to read input file");

    for (std::string currLn; std::getline(inStream, currLn);)
    {
        inputLines.push_back(currLn);
    }
}

bool rgbdBenchTUM::fetchData(cl_uint lineNumber)
{
    //Split line into parts divided by space
    std::vector<std::string> parts;
    boost::split(parts, inputLines[lineNumber], boost::is_any_of(" "));

    if (parts.size() != 12)
        return false;

    //parts now contains:
    //[0]:       CameraPos timestamp,
    //[1,2,3]:   Translation,
    //[4,5,6,7]: Rotation Quaternion,
    //[8]:       DepthMap timestamp,
    //[9]:       DepthFile name,
    //[10]:      ColorMap timestamp,
    //[11]:       ColorFile name

    //Load Depth according to http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    cv::Mat depthImg = cv::imread(inputFolder + parts[9], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    depthImg.convertTo(currDepth, CV_32FC1, (1.0 / 5000.0));

    //Load Color image and scale to [0..1]  
    cv::Mat colorImg = cv::imread(inputFolder + parts[11]);
    colorImg.convertTo(currColor, CV_32FC3, 1.0f / 255.0f);

    //Calculate camera position from translation and rotation quaternion
    currPos = Eigen::Matrix4f::Identity();
    currPos.topLeftCorner(3,3) = (new Eigen::Quaternionf(std::stof(parts[7]), std::stof(parts[4]), std::stof(parts[5]), std::stof(parts[6])))->toRotationMatrix();
    currPos.topRightCorner(3,1) = Eigen::Vector3f(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));

    //Save current line number
    currentLineNumber = lineNumber;

    return true;
}

cv::Mat rgbdBenchTUM::currentDepth()
{
    if (currentLineNumber >= inputLines.size()) throw std::logic_error("invalid line number");
    return currDepth;
}

cv::Mat rgbdBenchTUM::currentColor()
{
    if (currentLineNumber >= inputLines.size()) throw std::logic_error("invalid line number");
    return currColor;
}

Eigen::Matrix4f rgbdBenchTUM::currentPos()
{
    if (currentLineNumber >= inputLines.size()) throw std::logic_error("invalid line number");
    return currPos;
}

cl_uint rgbdBenchTUM::size()
{
    return inputLines.size();
}

cl_uint rgbdBenchTUM::currentLine()
{
    return currentLineNumber;
}

rgbdBenchTUM::~rgbdBenchTUM()
{
}
