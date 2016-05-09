#include "icp.hpp"

icp::icp(cl_float depthTrunc, Eigen::Matrix3f k) : cameraMatrix(cv::Mat1d(3, 3)), currentPos(Eigen::Matrix4f::Identity())
    
{
    cameraMatrix = cv::Mat::eye(3,3,CV_32FC1);
    eigen2cv(k, cameraMatrix);
    cvICP = cv::rgbd::Odometry::create("ICPOdometry");
    cvICP->setCameraMatrix(cameraMatrix);
}


void icp::calculateTransform(cv::Mat currColor, cv::Mat currDepth, cv::Mat nextColor, cv::Mat nextDepth)
{
    cv::Ptr<cv::rgbd::OdometryFrame> frame_prev = cv::Ptr<cv::rgbd::OdometryFrame>(new cv::rgbd::OdometryFrame()),
                                frame_curr = cv::Ptr<cv::rgbd::OdometryFrame>(new cv::rgbd::OdometryFrame());

    
    frame_curr->image = nextColor;
    frame_curr->depth = nextDepth;
    frame_prev->image = currColor;
    frame_prev->depth = currDepth;
    cv::Mat nextPos(4,4,CV_64FC1);
    cvICP->compute(frame_curr, frame_prev, nextPos);
    Eigen::Matrix4f mat = Eigen::Matrix4f();
    cv2eigen(nextPos, mat);
    Eigen::Matrix4f nextPosEigen(Eigen::Matrix4f::Identity());
    cv2eigen(nextPos, nextPosEigen);
    currentPos = currentPos * nextPosEigen;

    frame_prev->release();
    frame_curr->release();
}

Eigen::Matrix4f icp::getCurrPos()
{
    return currentPos;
}

icp::~icp()
{
}
