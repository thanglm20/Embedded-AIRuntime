/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/

#include "MnnFaceLandmark.hpp"
#include "faceAlign.hpp"
MnnFaceLandmark::MnnFaceLandmark(/* args */std::string mnn_path)
{
    this->ultraface_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = this->num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;

    this->ultraface_session = this->ultraface_interpreter->createSession(config);

    this->input_tensor = this->ultraface_interpreter->getSessionInput(this->ultraface_session, nullptr);
    printf("Init Mnn face landmark successfully\n");
}

MnnFaceLandmark::~MnnFaceLandmark()
{
    this->ultraface_interpreter->releaseModel();
    this->ultraface_interpreter->releaseSession(ultraface_session);
}

cv::Mat MnnFaceLandmark::executeMnnLandmark(const cv::Mat& img, int in_w, int in_h)
{
    
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
    }
    
    cv::Mat image;
    cv::resize(img, image, cv::Size(160, 160));

    this->ultraface_interpreter->resizeTensor(this->input_tensor, {1, 160, 160, 3});
    this->ultraface_interpreter->resizeSession(this->ultraface_session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(image.data, 160, 160, image.step[0], this->input_tensor);

    auto start = std::chrono::steady_clock::now();


    // run network
    ultraface_interpreter->runSession(this->ultraface_session);

    // get output data
    std::string Identity_2 = "Identity_2";
    MNN::Tensor *tensor_Identity_2 = this->ultraface_interpreter->getSessionOutput(this->ultraface_session, Identity_2.c_str());
    MNN::Tensor tensor_Identity_2_host(tensor_Identity_2, tensor_Identity_2->getDimensionType());
    tensor_Identity_2->copyToHostTensor(&tensor_Identity_2_host);


    std::map<std::string, MNN::Tensor*>  allTensor = this->ultraface_interpreter->getSessionOutputAll(this->ultraface_session);

    auto tensor_scores_host = new MNN::Tensor(tensor_Identity_2, MNN::Tensor::TENSORFLOW);//NHWC
   	tensor_Identity_2->copyToHostTensor(tensor_scores_host);
    
    // eye left
    float x_eye_left = ((tensor_Identity_2->host<float>()[76] + tensor_Identity_2->host<float>()[74]) / 2) * 160;
    float y_eye_left = ((tensor_Identity_2->host<float>()[77] + tensor_Identity_2->host<float>()[75]) / 2) * 160;
    // cv::Point pt4(x_eye_left, y_eye_left);
    // cv::circle(image, pt4, 3, cv::Scalar(0, 0, 255), -1 , 8, 0);
    
    // eye right
    float x_eye_right = ((tensor_Identity_2->host<float>()[88] + tensor_Identity_2->host<float>()[86]) / 2) * 160;
    float y_eye_right = ((tensor_Identity_2->host<float>()[89] + tensor_Identity_2->host<float>()[87]) / 2) * 160;
    // cv::Point pt5(x_eye_right, y_eye_right);
    // cv::circle(image, pt5, 3, cv::Scalar(0, 0, 255), -1 , 8, 0);
    
    // mount nose
    float x_nose = tensor_Identity_2->host<float>()[60] * 160;
    float y_nose = tensor_Identity_2->host<float>()[61] * 160;
    // cv::Point pt3(x_nose, y_nose);
    // cv::circle(image, pt3, 3, cv::Scalar(0, 0, 255), -1 , 8, 0);
    
    //mouth left
    float x_mouth_left = tensor_Identity_2->host<float>()[96] * 160;
    float y_mouth_left = tensor_Identity_2->host<float>()[97] * 160;
    // cv::Point pt1(x_mouth_left, y_mouth_left);
    // cv::circle(image, pt1, 3, cv::Scalar(0, 0, 255), -1 , 8, 0);
    
    // mount right
    float x_mouth_right = tensor_Identity_2->host<float>()[108] * 160;
    float y_mouth_right = tensor_Identity_2->host<float>()[109] * 160;
    // cv::Point pt2(x_mouth_right, y_mouth_right);
    // cv::circle(image, pt2, 3, cv::Scalar(0, 0, 255), -1 , 8, 0);
    
    cv::Mat src(5, 2, CV_32FC1, norm_face);
    float v2[5][2] =
			{ { x_eye_left, y_eye_left },
			{ x_eye_right, y_eye_right },
			{ x_nose, y_nose },
			{ x_mouth_left, y_mouth_left },
			{ x_mouth_right, y_mouth_right } };
    
    cv::Mat dst(5, 2, CV_32FC1, v2);
    cv::Mat m = similarTransform(dst, src);
	cv::Mat aligned(in_w, in_h, CV_32FC3);
	cv::Size size(in_w, in_h);

    //get aligned face with transformed matrix and resize to 112*112
	cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));		
	cv::warpAffine(image, aligned, transfer, size, 1, 0, 0);

    //cv::imwrite("/data/fr/landmark.jpg", aligned);
    printf("Execute Mnn face landmark successfully\n");
    return aligned;
}