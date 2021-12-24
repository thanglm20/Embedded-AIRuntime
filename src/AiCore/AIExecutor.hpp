
/*
    Module: AIExecutor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef AIExecutor_hpp
#define AIExecutor_hpp

#include "AITypeData.hpp"



class AIExecutor
{
private:
    /* data */
    airuntime::ExecutorType m_executor;
    airuntime::DeviceType m_device;
    airuntime::AlgTypeAI m_algType;
    std::string m_modelWeight;
    std::string m_modelParam;
    std::string m_pathLabel;

public:
    explicit AIExecutor(airuntime::ExecutorType executor,
                airuntime::DeviceType device,
                airuntime::AlgTypeAI algType,
                std::string pathLabel = "",
                std::string modelWeight = "",
                std::string modelParam = ""
                )
                :
                m_executor(executor),
                m_device(device),
                m_algType(algType),    
                m_pathLabel(pathLabel),            
                m_modelWeight(modelWeight),
                m_modelParam(modelParam)
                {}
                
    virtual ~AIExecutor() = default;

    virtual STATUS init() = 0;
    virtual STATUS run(const Mat& img, 
                        vector<ObjectTrace>& objects,
                        float threshold) = 0;
    virtual STATUS release() = 0;
    static AIExecutor* create(airuntime::ExecutorType executor, 
                            airuntime::DeviceType device,
                            airuntime::AlgTypeAI algType,
                            std::string pathLabel,
                            std::string modelWeight,
                            std::string modelParam = ""
                            );

    airuntime::DeviceType getDeviceType() {return m_device;}
    airuntime::AlgTypeAI getAlgType() {return m_algType;}
    std::string getModelWeight() {return m_modelWeight;}
    std::string getModelParam() {return m_modelParam;}
    std::string getPathLabel() {return m_pathLabel;}
};


#endif