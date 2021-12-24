/*
    Module: ColorClassifier.cpp
    Author: LE MANH THANG
    Created: Oct 30th, 2021
*/

#ifndef ColorClassifier_hpp
#define ColorClassifier_hpp

#include <iostream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/opencv.hpp>



class ColorClassifier
{
private:
    /* data */
    enum  m_Color { C_BLACK = 0, C_WHITE, C_GREY, C_RED, C_ORANGE, C_YELLOW, C_GREEN, C_AQUA, C_BLUE, C_PURPLE, C_PINK, NUM_COLOR_TYPES };
    m_Color GetPixelColorType(int H, int S, int V);
    m_Color GetPixelColorType(cv::Vec3b pixel);
public:
    std::vector<std::string> ColorNames = { "Black", "White", "Grey", "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Pink" };
    ColorClassifier(/* args */);
    ~ColorClassifier();
    std::string ClassifyColor(const cv::Mat& img);
    
};




#endif