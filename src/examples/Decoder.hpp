/*
    Module: Decoder.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#ifndef Decoder_hpp
#define Decoder_hpp

#include <iostream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


// #define USE_FFMPEG

class Decoder
{
public:

    virtual ~Decoder(){}
    virtual int open(const char* video_file) = 0;
    virtual cv::Mat getFrame() = 0;
};


#endif