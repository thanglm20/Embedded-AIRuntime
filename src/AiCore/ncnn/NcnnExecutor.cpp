/*
    Module: NcnnExecutor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#include "NcnnExecutor.hpp"


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
    cout << "Initiated NCNN Executor successfully\n";
    return STATUS::SUCCESS;
}

STATUS NcnnExecutor::run(const Mat& img, 
                            vector<ObjectTrace>& objects,
                            float threshold)
{

    return STATUS::SUCCESS;
}

STATUS NcnnExecutor::release(){

    cout << "Released NCNN Executor successfully\n";
    return STATUS::SUCCESS;
}


