/*
    Module: FrameManger.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#include "FrameManager.hpp"

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
    this->m_flNewFrame = true;
}
cv::Mat FrameManager::getFrame()
{
    this->m_flNewFrame = false;
    return this->m_frame;
}
bool FrameManager::isNewFrame()
{
    return this->m_flNewFrame;
}
float FrameManager::getFps()
{
    return this->m_fps;
}
unsigned long long FrameManager::getFrameCounter()
{
    return iFrameCounter;
}
