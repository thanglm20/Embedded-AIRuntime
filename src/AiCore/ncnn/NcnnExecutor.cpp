/*
    Module: NcnnExecutor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#include "NcnnExecutor.hpp"
#include "../utils/LoadLabel.hpp"

NcnnExecutor::NcnnExecutor(airuntime::DeviceType device, 
                            airuntime::AlgTypeAI algType,
                            std::string pathLabel,
                            std::string modelWeight, 
                            std::string modelParam
                            ) 
    : AIExecutor(airuntime::ExecutorType::NCNN, device, algType, pathLabel, modelWeight, modelParam)
    {
        cout << "==============> NCNN Executor <=================\n";
        cout << "Model weight: " << getModelWeight() << endl;
        cout << "Model param: " << getModelParam() << endl;
        cout << "Label: " << getPathLabel() << endl;
        cout << "================================================\n";
    }

NcnnExecutor::~NcnnExecutor()
{
    cout << "Called NcnnExecutor destructor\n";
}

STATUS NcnnExecutor::init(){
    if(getModelWeight().empty() || getPathLabel().empty() || getModelParam().empty())
    {
        cout << "Args are invalid, please check your input again\n";
        return STATUS::INVALID_ARGS;
    }
    if(getAlgType() == airuntime::AlgTypeAI::DETECT)
    {
        this->m_detector = new NcnnDetector();
        char pathModelWeight[100];
        sprintf(pathModelWeight, "%s", getModelWeight().c_str());
        char pathModelParam[100];
        sprintf(pathModelParam, "%s", getModelParam().c_str());
        int net = this->m_detector->initNcnnNetwork(pathModelWeight, pathModelParam, getDeviceType());
        if( net != STATUS::SUCCESS )
        {
            LOG_FAIL("Init Mobilenet SSD failed");
            return STATUS::FAIL;
        }
    }
    this->m_labels = loadObjectNames(getPathLabel());
    cout << "Initiated NCNN Executor successfully\n";
    return STATUS::SUCCESS;
}

STATUS NcnnExecutor::run(const Mat& img, 
                            vector<ObjectTrace>& objects,
                            float threshold)
{
    if(getAlgType() == airuntime::AlgTypeAI::DETECT)
    {
        this->m_detector->executeNcnnDetector(img, this->m_labels, objects, threshold);
    }
    return STATUS::SUCCESS;
}

STATUS NcnnExecutor::release(){

    cout << "Released NCNN Executor successfully\n";
    return STATUS::SUCCESS;
}


