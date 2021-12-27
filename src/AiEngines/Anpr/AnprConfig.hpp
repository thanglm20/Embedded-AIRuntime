/*
    Module: AnprConfig.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#ifndef AnprConfig_hpp
#define AnprConfig_hpp

#include <iostream>
#include "../../AiCore/AITypeData.hpp"

// #define ANDROID                                                                                                                                                                                                                                                                                                                                                                                                          

#define MAX_COUNT_UNKNOWN 10
#define THRESHOLD_ANPR 0.8
#define THRESHOLD_OCR 0.9
#define THRES_DETECT 0.5

using namespace std;
using namespace cv;

namespace airuntime{
    namespace aiengine{

enum class Nations {VN, US, MALAY};

struct PlateInfor
{
    cv::Mat imgPlate;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    cv::Rect rect;
    std::string license = "";
    std::string typePlate = "";                                                                                                                                                                                                             
    float score;
    int track_id = 0;
    bool isOutOfFrame = false;
    bool isNewEvent = false;
    uint8_t countUnknown = 0;
};
    }
}

#endif