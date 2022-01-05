/*
    Module: TrafficConfig.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#ifndef TrafficConfig_hpp
#define TrafficConfig_hpp

#include <iostream>
#include "../../AiCore/AITypeData.hpp"

// #define ANDROID                                                                                                                                                                                                                                                                                                                                                                                                          

#define THRESHOLD_VEHICLE 0.5

using namespace std;
using namespace cv;

namespace airuntime{
    namespace aiengine{
        namespace its{

struct VehicleTrace{
    cv::Rect rect;
    std::string License = "";
    std::string color = "";
    std::string type;
    float score;
    int track_id = 0;
    bool isOutOfFrame = false;
};

        }
    }
}

#endif