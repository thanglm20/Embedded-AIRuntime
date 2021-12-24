/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/

#ifndef MNNFACEDETECTOR_H
#define MNNFACEDETECTOR_H

#include "UltraFace.hpp"

class MnnFaceDetector
{
private:
    /* data */
    UltraFace* ultraface;
public:
    MnnFaceDetector(/* args */std::string pathModel);
    ~MnnFaceDetector();
    int initMnnFaceDetector(std::string pathModel);
    int executeMnnFaceDetector(const cv::Mat& img, std::vector<cv::Mat>& faces);
};


#endif