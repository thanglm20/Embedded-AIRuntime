/*
    Module: AIUserFactory.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#include "AIUserFactory.hpp"

#ifdef USE_SNPE
#include "snpe/SnpeExecutor.hpp"
#endif

#ifdef USE_NCNN
#include "ncnn/NcnnExecutor.hpp"
#endif

namespace airuntime{
namespace aicore
{

AIUserFactory::AIUserFactory(airuntime::ExecutorType executor,
                    airuntime::DeviceType device,
                    airuntime::AlgTypeAI algType,
                    std::string pathLabel,
                    std::string modelWeight,
                    std::string modelParam) {
    this->m_executor = AIExecutor::create(executor, device, algType, pathLabel, modelWeight, modelParam);
    this->m_flInitiated = false;

} 
   
AIUserFactory::~AIUserFactory()
{
    cout << "Called AIUserFactory destructor\n";
}



STATUS AIUserFactory::run(const Mat& img, 
                            vector<ObjectTrace>& objects,
                            float threshold)
{   
    if(!this->m_flInitiated)
    {
        if(this->m_executor == nullptr || STATUS::SUCCESS != this->m_executor->init())
        {
            cout << "Init AI User Factory failed\n";
            return STATUS::FAIL;
        }
        this->m_flInitiated = true;
    }
    if(img.empty() || threshold > 1.0 || threshold < 0.0)
    {
        cout << "Args are invalid, please check your input again\n";
        return STATUS::INVALID_ARGS;
    }
    objects.clear();
    this->m_executor->run(img, objects, threshold);
    return STATUS::SUCCESS;
}

STATUS AIUserFactory::release()
{
    this->m_flInitiated = false;
    if(this->m_executor) this->m_executor->release();
    return STATUS::SUCCESS;
}

}
}

AIExecutor* AIExecutor::create(airuntime::ExecutorType executor, 
                                airuntime::DeviceType device,
                                airuntime::AlgTypeAI algType,
                                std::string pathLabel,
                                std::string modelWeight,
                                std::string modelParam
                                )
{
    AIExecutor* exe = nullptr;
    switch ((int)executor)
    {
        #ifdef USE_SNPE
        case 0:
        {   
            exe = new SnpeExecutor(device, algType, pathLabel, modelWeight);
            break;
        }
        #endif

        #ifdef USE_NCNN
        case 1:
        {
            exe = new NcnnExecutor(device, algType, pathLabel, modelWeight, modelParam );
            break;
        }
        #endif

        default:
        {
            exe = nullptr;
            break;
        }  
    }
    return exe;
}
