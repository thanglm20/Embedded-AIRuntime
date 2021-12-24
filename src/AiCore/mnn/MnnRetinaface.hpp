/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#ifndef MNNRETINAFACE_H
#define MNNRETINAFACE_H

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

class MnnRetinaface
{
private:
    /* data */
    std::shared_ptr<MNN::Interpreter> retinaface_mnet25;
    MNN::Session *retinaface_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    int num_thread = 4;
    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};
    // const float mean_vals[3] = {1, 1, 1};
    // const float norm_vals[3] = {1, 1, 1};
public:
    MnnRetinaface(/* args */std::string mnn_path);
    ~MnnRetinaface();
    int executeMnnRetinaface(const cv::Mat& img);
};



#endif