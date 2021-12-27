/*
    Module: AnprDetector.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#ifndef AnprDetector_hpp
#define AnprDetector_hpp

#include "../../AiCore/AITypeData.hpp"
#include "../../AiCore/AIUserFactory.hpp"
#include "AnprConfig.hpp"



class AnprDetector
{
private:
    /* data */
    airuntime::aicore::AIUserFactory* m_plateDetector;
public:
    AnprDetector (airuntime::aiengine::Nations nation);
    ~AnprDetector ();
    int detect (cv::Mat& img, std::vector<ObjectTrace>& objects);
    
};

#endif