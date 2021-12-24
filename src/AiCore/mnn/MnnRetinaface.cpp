/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#include "MnnRetinaface.hpp"

MnnRetinaface::MnnRetinaface(/* args */std::string mnn_path)
{
    this->retinaface_mnet25 = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = this->num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;

    this->retinaface_session = this->retinaface_mnet25->createSession(config);

    this->input_tensor = this->retinaface_mnet25->getSessionInput(this->retinaface_session, nullptr);
    printf("Init Mnn Retinaface successfully\n"); 
}

MnnRetinaface::~MnnRetinaface()
{
    this->retinaface_mnet25->releaseModel();
    this->retinaface_mnet25->releaseSession(retinaface_session);
}

int MnnRetinaface::executeMnnRetinaface(const cv::Mat& img)
{
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
    }
    
    cv::Mat image;
    cv::resize(img, image, cv::Size(416, 288));

    this->retinaface_mnet25->resizeTensor(this->input_tensor, {1, 3, 288, 416});
    this->retinaface_mnet25->resizeSession(this->retinaface_session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
            MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
                                          norm_vals, 3));
    pretreat->convert(image.data, 416, 288, image.step[0], this->input_tensor);

    auto start = std::chrono::steady_clock::now();

    // run network
    retinaface_mnet25->runSession(this->retinaface_session);
    // face_rpn_bbox_pred_stride16 : 0xead77340
    // face_rpn_bbox_pred_stride32 : 0xead774c8
    // face_rpn_bbox_pred_stride8 : 0xead76ee0
    // face_rpn_cls_prob_reshape_stride16 : 0xead77148
    // face_rpn_cls_prob_reshape_stride32 : 0xead775a8
    // face_rpn_cls_prob_reshape_stride8 : 0xead77458
    // face_rpn_landmark_pred_stride16 : 0xead77378
    // face_rpn_landmark_pred_stride32 : 0xead77308
    // face_rpn_landmark_pred_stride8 : 0xead77490


    //std::map<std::string, MNN::Tensor*>  allTensor = this->retinaface_mnet25->getSessionOutputAll(this->retinaface_session);
    // std::string layer_name;
    // MNN::Tensor* tensor_out;
    // for( const auto& [layer_name, tensor_out] : allTensor)
    // {
    //      std::cout << layer_name << " : " << tensor_out << "\n";
    // }

    // get output data
    std::string Identity_2 = "face_rpn_bbox_pred_stride16";
    MNN::Tensor *tensor_Identity_2 = this->retinaface_mnet25->getSessionOutput(this->retinaface_session, Identity_2.c_str());
    MNN::Tensor tensor_Identity_2_host(tensor_Identity_2, tensor_Identity_2->getDimensionType());
    tensor_Identity_2->copyToHostTensor(&tensor_Identity_2_host);
    // for(int i = 0; i < tensor_Identity_2->size(); i++)
    // {
    //     float x_eye_left = tensor_Identity_2->host<float>()[i];
    //     printf("%f\n", x_eye_left);
    // }
    printf("Size of layer: %d\n", tensor_Identity_2->size());
    printf("Execute Mnn Retinaface successfully\n");
    return 0;
}
