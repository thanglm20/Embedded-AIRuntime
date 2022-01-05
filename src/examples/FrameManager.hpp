

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
    bool m_flNewFrame = false;
public:
    FrameManager();
    ~FrameManager();
    void updateFrame(const cv::Mat& frame);
    void updateFrame(const cv::Mat& frame, float fps);
    void clearFrame();
    cv::Mat getFrame();
    float getFps();
    unsigned long long getFrameCounter();
    bool isNewFrame();

};



#endif