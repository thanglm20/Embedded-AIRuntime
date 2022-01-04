/*
    Module: PaddleOCRx64.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#include "PaddleOCRx64.hpp"

PaddleOCRx64::PaddleOCRx64(){}

PaddleOCRx64::~PaddleOCRx64 () {}

STATUS PaddleOCRx64::init ()
{

    OCRConfig config(CONFIG_FILE);
    config.PrintConfigInfo();
    this->m_det = new DBDetector(config.det_model_dir, config.use_gpu, config.gpu_id,
                    config.gpu_mem, config.cpu_math_library_num_threads,
                    config.use_mkldnn, config.max_side_len, config.det_db_thresh,
                    config.det_db_box_thresh, config.det_db_unclip_ratio,
                    config.use_polygon_score, config.visualize,
                    config.use_tensorrt, config.use_fp16);

    this->m_cls = nullptr;
    if (config.use_angle_cls == true) {
        this->m_cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
                            config.gpu_mem, config.cpu_math_library_num_threads,
                            config.use_mkldnn, config.cls_thresh,
                            config.use_tensorrt, config.use_fp16);
    }

    this->m_rec = new CRNNRecognizer(config.rec_model_dir, config.use_gpu, config.gpu_id,
                        config.gpu_mem, config.cpu_math_library_num_threads,
                        config.use_mkldnn, config.char_list_file,
                        config.use_tensorrt, config.use_fp16);

    return STATUS::SUCCESS;
}
std::string PaddleOCRx64::recognize (cv::Mat& imgLicense)
{
    string text;
    std::vector<std::vector<std::vector<int>>> boxes;

    this->m_det->Run(imgLicense, boxes);

    text = this->m_rec->Run(boxes, imgLicense, this->m_cls);

    return text;
}
