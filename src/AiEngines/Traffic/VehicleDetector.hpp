/*
    Module: VehicleDetector.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef VehicleDetector_hpp
#define VehicleDetector_hpp


#include "../../AiCore/AIUserFactory.hpp"
#include "TrafficConfig.hpp"

class VehicleDetector
{
private:
    /* data */
    airuntime::aicore::AIUserFactory* m_executor;
public:
    VehicleDetector();
    ~VehicleDetector();
    STATUS run(const cv::Mat& img, std::vector<ObjectTrace>& objects, float threshold);
    STATUS release();
};

#endif