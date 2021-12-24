/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module:    NcnnDetector.hpp
 Author:    LE MANH THANG
 Created:   06/01/2021
 Modify:    HieuPV - 02/02/2021
 Description: 
********************************************************************************/

#ifndef NcnnDetector_H
#define NcnnDetector_H
#include "ncnn/net.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/videoio/videoio.hpp>
#endif

#include <vector>
#include <regex>
#include <fstream>
#include <stdio.h>

#define NCNN_PROFILING

#ifdef NCNN_PROFILING
#include "ncnn/benchmark.h"
#endif

#include "../AITypeData.hpp"
#include <unistd.h>


class NcnnDetector
{
    private:
        ncnn::Net* ncnnNet = nullptr;

        #ifdef ANDROID
        ncnn::UnlockedPoolAllocator* g_blob_pool_allocator_detect = nullptr;
        ncnn::PoolAllocator* g_workspace_pool_allocator_detect = nullptr;
        // HieuPV add code
        // Nen de la null_ptr nó khác với NULL(0)
        ncnn::VulkanDevice* g_vkdev = nullptr;
        ncnn::VkAllocator* g_blob_vkallocator = nullptr;
        ncnn::VkAllocator* g_staging_vkallocator = nullptr;
        #endif
        int width_model;
        int height_model;
    public:
        NcnnDetector();
        ~NcnnDetector();
        int initNcnnNetwork (const char* model_bin, const char* model_param);
        int initNcnnNetwork (const char* model_bin, const char* model_param, airuntime::DeviceType device);
        int executeNcnnDetector (const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects, float thres_detec) ;
};

#endif