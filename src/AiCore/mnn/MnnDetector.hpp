
/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: MnnDetector.hpp
 Author: LE MANH THANG
 Created: May 13th, 2021
 Description: 
********************************************************************************/
#ifndef MnnDetector_hpp
#define MnnDetector_hpp


#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

#include "AiTypeData.hpp"

using namespace std;
using namespace MNN;

#define hard_nms 1
#define blending_nms 2

struct BBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
};
const float mean_vals[3] = {0.f, 0.f, 0.f};
const float norm_vals[3] = {1/256.0f, 1/256.0f, 1/256.0f};
const float thres[2] = {0.6,0.5} ;
const int BIAS_W[6] = {26, 67, 72, 189, 137, 265};  
const int BIAS_H[6] = {48, 84, 175, 126, 236, 259};
const int INPUT_SIZE = 320;
const int CLASS_NUM = 20;
class MnnDetector
{
private:
    /* data */


    std::shared_ptr<MNN::Interpreter> interpreter;
    MNN::Session *session = nullptr;
    MNN::Tensor *input_tensor = nullptr;

public:
    MnnDetector(/* args */);
    ~MnnDetector();
    int init(std::string  path_model);
    int detect(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects, float thres_detect);
};


#endif