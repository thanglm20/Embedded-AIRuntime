

/*
    Module: BackgroundSubtractor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef BackgroundSubtractor_hpp
#define BackgroundSubtractor_hpp

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/video/background_segm.hpp>

#include "../ITS/TrafficManager.hpp"

#define NUMBER_FRAME_CATCH 100
#define MAX_FRAME_RESET 10000
class BackgroundSub
{
private:
    /* data */
    cv::Ptr<cv::BackgroundSubtractorMOG2> m_subMOG2;
    cv::Mat m_mask;
    int m_nFrameCounter = 0;
public:
    BackgroundSub(/* args */);
    ~BackgroundSub();
    void run(cv::Mat& img);
    void extract(cv::Mat& img);
    cv::Mat getBackground();
    void release();
};


#endif