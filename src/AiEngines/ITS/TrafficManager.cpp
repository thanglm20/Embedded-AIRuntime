

#include "TrafficManager.hpp"
#include "Violation/Oppose.hpp"

namespace airuntime{
    namespace aiengine{
        namespace its{


TrafficManager::TrafficManager()
{
    this->m_tracker = new ObjectTracking();
    this->m_executor = new airuntime::aicore::AIUserFactory(airuntime::ExecutorType::NCNN,
                                                            airuntime::DeviceType::CPU,
                                                            airuntime::AlgTypeAI::DETECT,
                                                            "../models/traffic.txt",
                                                            "../models/traffic.bin",
                                                            "../models/traffic.param");

    // this->m_executor = new airuntime::aicore::AIUserFactory(airuntime::ExecutorType::NCNN,
    //                                                         airuntime::DeviceType::CPU,
    //                                                         airuntime::AlgTypeAI::DETECT,
    //                                                         "../models/traffic.txt",
    //                                                         "../models/traffic.dlc");

}


TrafficManager::TrafficManager(TrafficSettings& settings)
{
    this->m_tracker = new ObjectTracking();
    this->m_executor = new airuntime::aicore::AIUserFactory(airuntime::ExecutorType::NCNN,
                                                            airuntime::DeviceType::CPU,
                                                            airuntime::AlgTypeAI::DETECT,
                                                            "../models/traffic.txt",
                                                            "../models/traffic.bin",
                                                            "../models/traffic.param");

    // this->m_executor = new airuntime::aicore::AIUserFactory(airuntime::ExecutorType::NCNN,
    //                                                         airuntime::DeviceType::CPU,
    //                                                         airuntime::AlgTypeAI::DETECT,
    //                                                         "../models/traffic.txt",
    //                                                         "../models/traffic.dlc");

    this->m_settings = settings;
    settings.violation.oppose.isUsed = true;
    if(settings.violation.oppose.isUsed)
    {
        std::cout << "=============== Setting oppose ================\n";
        this->m_violation = new Oppose(settings.violation);
    }
    
}


TrafficManager::~TrafficManager()
{

}


STATUS TrafficManager::run(const cv::Mat& img, std::vector<VehicleTrace>& output)
{
    
    std::vector<ObjectTrace> objects;
    if(STATUS::SUCCESS != this->m_executor->run(img, objects, THRESHOLD_VEHICLE))
        return STATUS::FAIL;
    else
    {
        // process tracking
        std::vector<TrackingTrace> tracks;
        this->m_tracker->process(objects, tracks);

        if(this->m_settings.violation.oppose.isUsed)
            this->m_violation->process(img, this->m_tracker, tracks);
        // //delete object which is abandoned
        // for(auto it = this->m_listVehicles.begin(); it != this->m_listVehicles.end();)
        // {
        //     const int theId =  (*it).track_id;
        //     const auto p = find_if(tracks.begin(), tracks.end(), 
        //                                 [theId] ( const TrackingTrace& a ) { return (a.m_ID == theId);}); 
        //     if (p == tracks.end() && it != this->m_listVehicles.end())
        //         it = this->m_listVehicles.erase(it);                
        //     else 
        //         it++;
        // }
        // process new track and update old track
        for(auto track : tracks)
        {
            
            // if(!track.isOutOfFrame)
            // {
            //     VehicleTrace vehicle;
            //     vehicle.rect = track.m_rect;
            //     vehicle.track_id = track.m_ID;
            //     vehicle.type = track.m_type;
            //     output.push_back(vehicle);
            // }
            // // else
            // // {
            // //     const int theId =  track.m_ID;
            // //     const auto p = find_if(this->m_listVehicles.begin(), this->m_listVehicles.end(), 
            // //                             [theId] ( const VehicleTrace& a ) { return (a.track_id == theId);});                         
            // //     if (p != this->m_listVehicles.end()) 
            // //     {
            // //         int dist = distance(this->m_listVehicles.begin(), p);
            // //         this->m_listVehicles[dist].isOutOfFrame = true;
                    
            // //     }
            // // } 
        }

        //get list of output plates
        // for(auto p : this->m_listVehicles)
        //     if(p.isOutOfFrame != true) 
        //         output.push_back(p);                         
    }
    
    return STATUS::SUCCESS;
}

STATUS TrafficManager::release()
{

}

        } // namespace its
    } // namespace engine
} // namespace airuntime