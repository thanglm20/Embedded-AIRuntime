/******************************************************************************** 
 Module: ObjectDetector.cpp
 Author: LE MANH THANG
 Created: 21/12/2020
 Description: 
********************************************************************************/
#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include "snpe/detector/SnpeDetector.hpp"
#include "ncnn/NcnnDetector.hpp"
// #include "mnn/MnnDetector.hpp"
// #include "tflite/TfliteDetector.hpp"
#include "AiTypeData.hpp"
#include <regex>
#include <unistd.h>
class ObjectDetector
{
    private:
        std::vector<std::string> labels;
        std::string framework;
        std::string objectTarget;
        // Khởi tạo tất cả là null_ptr
        // TfliteDetector* tflite_detector = nullptr;
        // MnnDetector* mnn_detector = nullptr;
        SnpeDetector* snpe_detector = nullptr;
        NcnnDetector* ncnn_detector = nullptr;
       
    public:
        ObjectDetector();
        ~ObjectDetector();
        int initObjectDetector(std::string framework, std::string objectTarget);
        int initObjectDetector(std::string framework, std::string objectTarget, std::string deviceTarget);
        int executeObjectDetector(const cv::Mat& img, std::vector<ObjectTrace>& objects, float thres_detect);
};

#endif