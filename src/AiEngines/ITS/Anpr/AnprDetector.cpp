/*
    Module: AnprDetector.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#include "AnprDetector.hpp"



AnprDetector::AnprDetector(airuntime::aiengine::Nations nation)
{   
    #ifdef ANDROID
    if(nation == airuntime::aiengine::Nations::VN)
        this->m_plateDetector = new airuntime::aicore::AIUserFactory(
                                            airuntime::ExecutorType::SNPE, 
                                            airuntime::DeviceType::DSP,
                                            airuntime::AlgTypeAI::DETECT,
                                            "/data/thanglmb/models/snpe/AnprDetect.txt",
                                            "/data/thanglmb/models/snpe/AnprDetect.dlc"
                                            );
    #else
        if(nation == airuntime::aiengine::Nations::VN)
        this->m_plateDetector = new airuntime::aicore::AIUserFactory(
                                            airuntime::ExecutorType::NCNN, 
                                            airuntime::DeviceType::CPU,
                                            airuntime::AlgTypeAI::DETECT,
                                            "../models/anpr/AnprDetect.txt",
                                            "../models/anpr/AnprDetect.bin",
                                            "../models/anpr/AnprDetect.param"
                                            );
    #endif
}

AnprDetector::~AnprDetector()
{
    if(this->m_plateDetector) 
        this->m_plateDetector->release();
}



int AnprDetector::detect (cv::Mat& img, std::vector<ObjectTrace>& objects)
{
    if(!img.empty())
    {
        #if(VEHICLE_DETECTION == 1)
        cv::Mat srcImg;
        img.copyTo(srcImg);
        objects.clear();
        std::vector<ObjectTrace> vehicles;
        this->vehicle->detect(img, vehicles);
        std::cout << "Number of vehicles: " << vehicles.size() << std::endl;
        for(int i = 0; i < vehicles.size(); i++)
        {

            cv::Mat imgVehicle = img(vehicles[i].rect).clone();
            float w_ratio = img.cols / imgVehicle.cols;
            float h_ratio = img.rows / imgVehicle.rows;
            
            std::vector<ObjectTrace> plates;
            this->obj_detector->executeObjectDetector(imgVehicle, plates, 0.3);
            for(int j = 0; j < plates.size(); j++)
            {
                ObjectTrace plate = plates[j];
                plate.rect.x = vehicles[i].rect.x + plate.rect.x;
                plate.rect.y = vehicles[i].rect.y +  plate.rect.y;
                plate.rect.width = plate.rect.width;
                plate.rect.height = plate.rect.height;
                objects.push_back(plate);
            }
            cv::rectangle(srcImg, vehicles[i].rect, cv::Scalar(255, 0, 0), 1, 8);
        }
        
        #else 
            if(this->m_plateDetector->run(img, objects, THRES_DETECT) != STATUS::SUCCESS)
            {
                LOG_FAIL("Execute Anpr detector faield");
                return STATUS::FAIL;
            }
        #endif
    }
    else 
    {
        LOG_FAIL("Executing Anpr detector failed, please check your input");
        return STATUS::INVALID_ARGS;
    }
    return STATUS::SUCCESS;
}


