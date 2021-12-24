/*
    Module: SnpeExecutor.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef SnpeExecutor_hpp
#define SnpeExecutor_hpp

#include "../AIExecutor.hpp"
#include "SnpeMobilenetSSD.hpp"

class SnpeExecutor : public AIExecutor
{
private:
    /* data */
    SnpeMobilenetSSD* m_detector = nullptr;
    vector<string> m_labels;
public:
    explicit SnpeExecutor(airuntime::DeviceType device, 
                            airuntime::AlgTypeAI algType,
                            std::string pathLabel,
                            std::string modelWeight
                            );
    ~SnpeExecutor();
    virtual STATUS init() override;
    virtual STATUS run(const Mat& img, 
                        vector<ObjectTrace>& objects,
                        float threshold) override;
    virtual STATUS release() override;
};





#endif