/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeRuntime.cpp
 Author: LE MANH THANG
 Created: 08/02/2021
 Description: 
********************************************************************************/
#ifndef SNPECLASSIFIER_H
#define SNPECLASSIFIER_H
#include "SnpeCommLib.hpp"
#include "AiTypeData.hpp"
class SnpeClassifier
{
private:
    /* data */
    std::unique_ptr<zdl::SNPE::SNPE> snpeClassifer;
public:
    SnpeClassifier(/* args */);
    ~SnpeClassifier();
    int initSnpeClassifier(std::string containerPath);
    int executeSnpeClassifier(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects);

};
#endif