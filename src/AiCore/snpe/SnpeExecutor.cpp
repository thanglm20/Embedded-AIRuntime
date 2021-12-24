/*
    Module: SnpeExecutor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#include "SnpeExecutor.hpp"
#include "../utils/LoadLabel.hpp"

SnpeExecutor::SnpeExecutor(airuntime::DeviceType device, 
                            airuntime::AlgTypeAI algType,
                            std::string pathLabel,
                            std::string modelWeight
                            ) 
    : AIExecutor(airuntime::ExecutorType::SNPE, device, algType, pathLabel, modelWeight)
    {
        cout << "==============> SNPE Executor <=================\n";
        cout << "Model weight: " << getModelWeight() << endl;
        cout << "Model param: " << getModelParam() << endl;
        cout << "Label: " << getPathLabel() << endl;
        cout << "================================================\n";
    }

SnpeExecutor::~SnpeExecutor()
{
    cout << "Called SnpeExecutor destructor\n";
}

STATUS SnpeExecutor::init()
{
    SetAdspLibraryPath(SNPE_LIB_PATH);
    if(getModelWeight().empty() || getPathLabel().empty())
    {
        cout << "Args are invalid, please check your input again\n";
        return STATUS::INVALID_ARGS;
    }
    if(getAlgType() == airuntime::AlgTypeAI::DETECT)
    {
        this->m_detector = new SnpeMobilenetSSD();
        int net = this->m_detector->initSnpeMobilenetSSD(getModelWeight(), getDeviceType());
        if( net != STATUS::SUCCESS )
        {
            LOG_FAIL("Init Mobilenet SSD failed");
            return STATUS::FAIL;
        } 
    }
    this->m_labels = loadObjectNames(getPathLabel());
    cout << "Initiated SNPE Executor successfully\n";
    return STATUS::SUCCESS;
}

STATUS SnpeExecutor::run(const Mat& img, 
                            vector<ObjectTrace>& objects,
                            float threshold)
{
    if(getAlgType() == airuntime::AlgTypeAI::DETECT)
    {
        this->m_detector->executeSnpeMobilenetSSD(img, this->m_labels, objects, threshold);
    }
    return STATUS::SUCCESS;
}

STATUS SnpeExecutor::release(){

    if(this->m_detector) delete this->m_detector;
    cout << "Released SNPE Executor successfully\n";
    return STATUS::SUCCESS;
}

