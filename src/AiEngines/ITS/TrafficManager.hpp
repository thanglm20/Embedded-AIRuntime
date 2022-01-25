/*
    Module: TrafficManager.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef TrafficManager_hpp
#define TrafficManager_hpp


#include "../../AiCore/AIUserFactory.hpp"
#include "../../AiCore/sort-tracker/ObjectTracking.hpp"
#include "Violation/Violation.hpp"
#include "Anpr/AnprRecognizer.hpp"

#define THRESHOLD_VEHICLE 0.5

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

struct TrafficSettings{
    bool isUsedAnpr = false;
    ViolationSettings violation;
};

class TrafficManager
{
private:
    /* data */
    TrafficSettings m_settings;
    // std::vector<std::unique_ptr<Violation> m_supervisor; 
    Violation* m_violation;
    AnprRecognizer* m_anpr;
    airuntime::aicore::AIUserFactory* m_executor = nullptr;
    ObjectTracking* m_tracker = nullptr;
    std::vector<VehicleTrace> m_listVehicles;
public:
    TrafficManager();
    TrafficManager(TrafficSettings& settings);
    ~TrafficManager();
    STATUS run(const cv::Mat& img, std::vector<VehicleTrace>& output);
    STATUS release();
};
        }
    }
}
#endif