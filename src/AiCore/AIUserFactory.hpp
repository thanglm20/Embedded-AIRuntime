/*
    Module: AIUserFactory.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef AIUserFactory_hpp
#define AIUserFactory_hpp

#include "AIExecutor.hpp"

namespace airuntime{
namespace aicore{
    
class AIUserFactory
{
private:
    /* data */
    AIExecutor* m_executor;
    bool m_flInitiated = false;
public:
    explicit AIUserFactory(airuntime::ExecutorType executor,
                    airuntime::DeviceType device,
                    airuntime::AlgTypeAI algType,
                    std::string pathLabel,
                    std::string modelWeight,
                    std::string modelParam = ""
                    );           
    ~AIUserFactory();
    
    STATUS run(const Mat& img, 
                vector<ObjectTrace>& objects,
                float threshold);
    STATUS release();
};
}
}

#endif
