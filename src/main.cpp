#include <iostream>
#include <vector>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/rgbd.hpp>
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#include "math.hpp"
#include "tsdf.hpp"
#include "OpenclFrame.hpp"
#include "clBuffer.hpp"
#include "kernelExecutors.hpp"
#include "helpers.hpp"
#include "raycaster.hpp"
#include "rgbdBenchTUM.hpp"
#include "paramProvider.hpp"
#include "icp.hpp"
#include "timer.hpp"

#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <exception>

#define FRAME_TIME                    33  //in milliseconds, set to <= 0 to disable
#define KINECT                    0   //1 use kinect, 0 use files
#if KINECT
# define CX  319.5f
# define CY  239.5f 
# define FX  525.0f
# define FY  525.0f
#else
# define CX  318.6f
# define CY  255.3f 
# define FX  517.3f
# define FY  516.5f
#endif


void outputVolume(const float* volume, size_t w, size_t h, size_t d, const char* name)
{
  cv::Mat mSlice(h, w, CV_32FC1);
  for (size_t z = 0; z < d; ++z) {
    const float* start = volume + (w * h * z);
    float maxValue;
    float minValue;
    calcMinMax(start, w, h, 1, minValue, maxValue);
    //output
    interleave(mSlice, start);
    if (minValue == maxValue)
        mSlice = 0;
    else
        mSlice = (mSlice - minValue) / (maxValue - minValue);
    std::stringstream ss;
    ss << PROJECT_DIR << "/slices/" << name << "_" << z << ".png";
    cv::imwrite(ss.str(), mSlice * 255.f);
  }
  std::cout << "printed slices for " << name << std::endl;
}


int main(int argc, char * argv[])
{
    //Read parameters
    try
    {
        //Parse command line parameters
        paramProvider paramProv(argc, argv);
        
        //Initialize intrinsic camera matirx
        Eigen::Matrix3f k;
        Eigen::Matrix3f kInv;
        k <<    FX, 0.0, CX,
                0.0, FY, CY,
                0.0, 0.0, 1.0;
        kInv = Eigen::Matrix3f(k.inverse());

        //Set up provider object for benchmark data
        rgbdBenchTUM benchProvider(paramProv.inputPath(), "rgbd_assoc_poses.txt");

        //Init OpenCL context
        OpenclFrame clFrame;
        
        //Initialize timers
        timer frameTime(clFrame);
        timer tsdfIntegrationTime(clFrame);
        timer raycastTime(clFrame);
        timer icpTime(clFrame);
        
        //load first frame to determine width and height of our images and initial pose
        benchProvider.fetchData(0);
        cl_uint w = benchProvider.currentDepth().cols;
        cl_uint h = benchProvider.currentDepth().rows;
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f initialPose = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f initialPoseInv = Eigen::Matrix4f::Identity();

         //Init ICP
        icp icpInstance(paramProv.depthTruncation(), k);

        if (paramProv.isUseICP())
        {
            //Use groundtruth pose
            initialPose = benchProvider.currentPos();
            initialPoseInv = initialPose.inverse();
        }

        //Init raycaster
        raycaster rc(clFrame, w, h, k);

        //Create buffers
        clBuffer<cl_float> depth(clFrame, w*h);
        clBuffer<cl_float> colorImage(clFrame, w*h*3);
        clBuffer<cl_float> vertexMap(clFrame, w*h*3);
        clBuffer<cl_float> normalMap(clFrame, w*h*3);
        clBuffer<cl_float> raycastVertexMap(clFrame, w*h*3);    
        clBuffer<cl_float> raycastNormalMap(clFrame, w*h*3);    
        clBuffer<cl_float> raycastColorMap(clFrame, w*h*3);
        tsdf tsdfInstance(clFrame, paramProv.volumeRes(), paramProv.volumeSize());

        //Fuse every frame into TSDF
        Eigen::Vector3f centroid;
        centroid << 0.f, 0.f, 0.f;
        Eigen::Matrix4f centroidTrans = Eigen::Matrix4f(Eigen::Matrix4f::Identity());

        //Define variables for icp
        cv::Mat raycastDepth(h, w, CV_32FC1);
        cv::Mat raycastColor(h, w, CV_32FC3);
        cv::Mat nextDepth(h, w, CV_32FC1);
        cv::Mat nextColor(h, w, CV_32FC3);  

        //Define maximum number of frames
        cl_uint maxNumFrames = benchProvider.size();

        for (cl_uint frameNumber = 0; frameNumber < benchProvider.size() && frameNumber < maxNumFrames; ++frameNumber)
        {
            //Start timer for entire frame
            frameTime.start();
            
            //Fetch current frame
            benchProvider.fetchData(frameNumber);
            
            //Copy depth and color to buffer
            layer(depth.hostBuffer(), benchProvider.currentDepth());
            depth.upload();
            layer(colorImage.hostBuffer(), benchProvider.currentColor());
            colorImage.upload();

            //Run vertexmap Kernel (Sets invalid depth to nan which is needed by ICP algorithm in O)penCV
            calculateVertexMap(clFrame, vertexMap.deviceBuffer(), depth.deviceBuffer(), convertEigen(kInv), paramProv.depthTruncation(), (cl_uint)w, (cl_uint)h);

            //Calculate centroid of first frame fo center volume around desired object
            if (frameNumber == 0) {
                vertexMap.download();
                Eigen::Vector3f cent = convertEigen(calculateCentroid(vertexMap.hostBuffer(), w, h));
                centroidTrans.topRightCorner<3, 1>() = -cent;
            }

            //Find new pose
            if (paramProv.isUseICP())
            {
                //Start ICP timer
                icpTime.start();
                //Depth with nan for invalid values
                depth.download();
                interleave(nextDepth, depth.hostBuffer());

                cvtColor(benchProvider.currentColor(), nextColor, cv::COLOR_BGR2GRAY);
                //Approximate new pose using ICP
                if (frameNumber > 0)
                {                    
                    icpInstance.calculateTransform(raycastColor, raycastDepth, nextColor, nextDepth);
                }

                pose = icpInstance.getCurrPos();
                //Stop ICP timer
                icpTime.end();
            }
            else
            {
                pose = initialPoseInv * benchProvider.currentPos();
            }

            //calculate current pose
            Eigen::Matrix4f trans = centroidTrans * pose;

            //Run normalmap Kernel
            calculateNormalMap(clFrame, normalMap.deviceBuffer(), vertexMap.deviceBuffer(), paramProv.normalDerivTrunc(), w, h);

            //Start tsdf integration timer
            tsdfIntegrationTime.start();
            //Integrate frame into tsdf volume
            tsdfInstance.integrate(depth.deviceBuffer(), colorImage.deviceBuffer(), k, Eigen::Matrix4f(trans.inverse()), paramProv.tsdfMaxTrunc(), paramProv.tsdfMinTrunc(), 1 << 7, w, h, paramProv.volumeRes(), paramProv.volumeSize());
            //Stop tsdf integration timer
            tsdfIntegrationTime.end();

            //Start raycast timer
            raycastTime.start();
            //Raycast volume
            rc.raycastVolume(tsdfInstance, trans);
            //Stop raycast timer
            raycastTime.end();

            //Output
            normalMap.download();
            show3DData(normalMap.hostBuffer(), w, h, "Normals", 100, 50);

            rc.normalMapBuffer().download();
            rc.normalMapBuffer().hostBuffer();
            show3DData(rc.normalMapBuffer().hostBuffer(), w, h, "RaycastNormals", 100, 50+h);

            rc.colorMapBuffer().download();
            rc.colorMapBuffer().hostBuffer();
            show3DData(rc.colorMapBuffer().hostBuffer(), w, h, "Raycast Colors", 100+w, 50);

            rc.vertexMapBuffer().download();
            rc.vertexMapBuffer().hostBuffer();
            show3DData(rc.vertexMapBuffer().hostBuffer(), w, h, "Raycast Vertices", 100+w*2, 50);

            rc.depthMapBuffer().download();
            rc.depthMapBuffer().hostBuffer();
            displayImage(raycastDepth, "Raycast Depth", 100+w, 50+h);

            displayImage(benchProvider.currentDepth(), "Depth", 100+w*2, 50+h);
            cv::waitKey(FRAME_TIME);

            //Set raycast depth and color for next frame
            
            interleave(raycastDepth, rc.depthMapBuffer().hostBuffer());
            interleave(raycastColor, rc.colorMapBuffer().hostBuffer());
            std::cout << "Frame: " << frameNumber << " integrated" << std::endl;

            //Stop timer for entire frame
            frameTime.end();

            //Print timings
            std::cout << "Frame time: " <<  frameTime.measurement() * 1000.f << "ms" << std::endl;
            std::cout << "TSDF integration time: " <<  tsdfIntegrationTime.measurement() * 1000.f << "ms" << std::endl;
            if (paramProv.isUseICP())
                std::cout << "ICP time: " <<  icpTime.measurement() * 1000.f << "ms" << std::endl;
            std::cout << "Raycast time: " <<  raycastTime.measurement() * 1000.f << "ms" << std::endl;    
        }
        tsdfInstance.valuesBuffer().download();
        tsdfInstance.colorsBuffer().download();
        //Print volume slices
        outputVolume(tsdfInstance.valuesBuffer().hostBuffer(), paramProv.volumeRes(), paramProv.volumeRes(), paramProv.volumeRes(), "TSDFValues");
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }
}
