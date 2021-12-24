/*
    Module: NcnnExecutor.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef NcnnExecutor_hpp
#define NcnnExecutor_hpp


#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <unistd.h>
#include <vector>
#include <regex>
#include <fstream>
#include <stdio.h>

#include "ncnn/net.h"

#define NCNN_PROFILING

#ifdef NCNN_PROFILING
#include "ncnn/benchmark.h"
#endif

#include "../AITypeData.hpp"
#include "../AIExecutor.hpp"

class NcnnExecutor : public AIExecutor
{
private:
    /* data */
    ncnn::Net* ncnnNet = nullptr;
    ncnn::UnlockedPoolAllocator* g_blob_pool_allocator_detect = nullptr;
    ncnn::PoolAllocator* g_workspace_pool_allocator_detect = nullptr;
    // HieuPV add code
    // Nen de la null_ptr nó khác với NULL(0)
    ncnn::VulkanDevice* g_vkdev = nullptr;
    ncnn::VkAllocator* g_blob_vkallocator = nullptr;
    ncnn::VkAllocator* g_staging_vkallocator = nullptr;
    int width_model;
    int height_model;
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