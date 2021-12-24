/******************************************************************************** 
 Module: SnpeRuntime.cpp
 Author: LE MANH THANG
 Created: 08/02/2021
 Description: 
********************************************************************************/
#ifndef  SNPEMOBILENETSSD_H
#define SNPEMOBILENETSSD_H

#include "SnpeCommLib.hpp"
#include "../AITypeData.hpp"



class SnpeMobilenetSSD
{
private:
    /* data */
    snpeBuilders* snpeMobilenetSSD;
public:
    SnpeMobilenetSSD(/* args */);
    ~SnpeMobilenetSSD();
    int initSnpeMobilenetSSD(std::string containerPath);
    int initSnpeMobilenetSSD(std::string containerPath, airuntime::DeviceType device);
    int executeSnpeMobilenetSSD(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects, float thres_detect);
    int executeSnpeMobilenetSSD(const uint8_t* rawData, int w, int h, std::vector<std::string>& labels, 
                                std::vector<ObjectTrace>& objects, float thres_detect);
};

#endif