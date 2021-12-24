/*
    Module: AnprDetector.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#ifndef AnprDetector_hpp
#define AnprDetector_hpp

#include "../../AiCore/AITypeData.hpp"
#include "../../AiCore/AIUserFactory.hpp"


#define THRES_DETECT 0.5
enum class Nations {VN ,VnRect, VnSquare, US, MALAY };

class AnprDetector
{
private:
    /* data */
    airuntime::aicore::AIUserFactory* m_plateDetector;
public:
    AnprDetector (Nations nation);
    ~AnprDetector ();
    int detect (cv::Mat& img, std::vector<ObjectTrace>& objects);
    
};

#endif