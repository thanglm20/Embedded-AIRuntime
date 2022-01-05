/*
    Module: TrafficManager.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef TrafficManager_hpp
#define TrafficManager_hpp


#include "../../AiCore/AIUserFactory.hpp"
#include "../../AiCore/sort-tracker/ObjectTracking.hpp"
#include "TrafficConfig.hpp"



namespace airuntime{
    namespace aiengine{
        namespace its{
            
class TrafficManager
{
private:
    /* data */
    airuntime::aicore::AIUserFactory* m_executor = nullptr;
    ObjectTracking* m_tracker = nullptr;
    std::vector<VehicleTrace> m_listVehicles;
public:
    TrafficManager();
    ~TrafficManager();
    STATUS run(const cv::Mat& img, std::vector<VehicleTrace>& output);
    STATUS release();
};
        }
    }
}
#endif