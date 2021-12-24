/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/

#ifndef MNNFACELANDMARK_H
#define MNNFACELANDMARK_H


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

class MnnFaceLandmark
{
private:
    /* data */
    std::shared_ptr<MNN::Interpreter> ultraface_interpreter;
    MNN::Session *ultraface_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    int num_thread = 4;
    // const float mean_vals[3] = {127, 127, 127};
    // const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};
    const float mean_vals[3] = {1, 1, 1};
    const float norm_vals[3] = {1, 1, 1};
public:
    MnnFaceLandmark(/* args */std::string mnn_path);
    ~MnnFaceLandmark();
    cv::Mat executeMnnLandmark(const cv::Mat& img, int in_w, int in_h);
};



#endif