

#include "VehicleDetector.hpp"


VehicleDetector::VehicleDetector()
{
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
VehicleDetector::~VehicleDetector()
{

}


STATUS VehicleDetector::run(const cv::Mat& img, std::vector<ObjectTrace>& objects, float threshold)
{
    
    if(!img.empty()) 
    {   
        if(STATUS::SUCCESS != this->m_executor->run(img, objects, threshold))
        return STATUS::FAIL;
    }
    return STATUS::SUCCESS;
    
}

STATUS VehicleDetector::release()
{

}