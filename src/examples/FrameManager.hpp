

/*
    Module: FrameManger.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#ifndef FrameManager_hpp
#define FrameManager_hpp

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

class FrameManager
{
private:
    /* data */
    cv::Mat m_frame;
    float m_fps = 0;
    static unsigned long long iFrameCounter;
public:
    FrameManager();
    ~FrameManager();
    void updateFrame(const cv::Mat& frame);
    void updateFrame(const cv::Mat& frame, float fps);
    cv::Mat getFrame();
    float getFps();
    unsigned long long getFrameCounter();
};

unsigned long long FrameManager::iFrameCounter = 0;

FrameManager::FrameManager(){
    iFrameCounter = 0;
    this->m_fps = 0;
}
FrameManager::~FrameManager(){;}

void FrameManager::updateFrame(const cv::Mat& frame)
{
    frame.copyTo(this->m_frame);
    iFrameCounter++;
    if(iFrameCounter == ULONG_LONG_MAX)
        iFrameCounter = 0;
}

void FrameManager::updateFrame(const cv::Mat& frame, float fps)
{
    frame.copyTo(this->m_frame);
    this->m_fps = fps;
    iFrameCounter++;
    if(iFrameCounter == ULONG_LONG_MAX)
        iFrameCounter = 0;
}
cv::Mat FrameManager::getFrame()
{
    return this->m_frame;
}
float FrameManager::getFps()
{
    return this->m_fps;
}
unsigned long long FrameManager::getFrameCounter()
{
    return iFrameCounter;
}

#endif