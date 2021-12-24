/*
    Module: NcnnExecutor.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef NcnnExecutor_hpp
#define NcnnExecutor_hpp

#include "../AIExecutor.hpp"

class NcnnExecutor : public AIExecutor
{
private:
    /* data */
public:
    explicit NcnnExecutor(airuntime::DeviceType device, 
                        airuntime::AlgTypeAI algType,
                        std::string pathLabel,
                        std::string modelWeight, 
                        std::string modelParam
                        );
    ~NcnnExecutor();
    virtual STATUS init() override;
    virtual STATUS run(const Mat& img, 
                        vector<ObjectTrace>& objects,
                        float threshold) override;
    virtual STATUS release() override;
};





#endif