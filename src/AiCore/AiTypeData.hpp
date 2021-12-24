
/******************************************************************************* 
 Module: 
 Author: LE MANH THANG
 Created: 21/12/2020
 Description: 
********************************************************************************/
#ifndef CONFIG_H
#define CONFIG_H

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <chrono> 

using namespace std::chrono; 

#define PATH_DICTIONARY "/data/thanglmb/models/paddle/anpr.txt"

#define DIR_OCR_DET_VN "/data/thanglmb/models/paddle/ocr_detect_vn.nb" 
#define DIR_OCR_RECOG_VN "/data/thanglmb/models/paddle/ocr_recog_vn.nb"

#define DIR_OCR_DET_US "/data/thanglmb/models/paddle/us-det.nb" 
#define DIR_OCR_RECOG_US "/data/thanglmb/models/paddle/us-rec.nb"

#define DIR_OCR_DET_MALAY "/data/thanglmb/models/malay-det.nb"
#define DIR_OCR_RECOG_MALAY "/data/thanglmb/models/malay-rec.nb"

#define DIR_MODEL_OBJECTDETECION_NCNN "/media/thanglmb/Bkav/AICAM/AIProject/AiCorex86_64/models/" // G1
#define DIR_LABEL_OBJECTDETECION_NCNN "/media/thanglmb/Bkav/AICAM/AIProject/AiCorex86_64/models/" // G1

#define DIR_MODEL_OBJECTDETECION_SNPE "/media/thanglmb/Bkav/AICAM/AIProject/AiCorex86_64/models/" // G2
#define DIR_LABEL_OBJECTDETECION_SNPE "/media/thanglmb/Bkav/AICAM/AIProject/AiCorex86_64/models/" // G2

#define DIR_MODEL_OBJECTCLASSIFICATION "/system/app/ai_config/models/snpe/" //G2
#define DIR_LABEL_OBJECTCLASSIFICATION "/system/app/ai_config/models/snpe/" //G2

#define DIR_MODEL_OBJECTCLASSIFICATION_TFLITE "/system/app/ai_config/models/tflite/multilabel.tflite"
#define DIR_LABEL_OBJECTDETECION_TFLITE "/system/app/ai_config/models/tflite/" // G2
#define DIR_MODEL_OBJECTDETECION_TFLITE "/system/app/ai_config/models/tflite/" 

#define DIR_MODEL_OBJECTDETECION_MNN "/system/app/ai_config/models/mnn/"
#define DIR_LABEL_OBJECTDETECION_MNN "/system/app/ai_config/models/mnn/"

struct ObjectTrace
{
    cv::Rect2f rect; // xmin, ymin, width, height
    std::string label;
    float score;
    int obj_id;   
};

#define STATUS_FAILED -1
#define STATUS_SUCCESS 0

#define LOG_INFO(log) std::cout << "[INFO] - " << log << std::endl;
#define LOG_SUCCESS(log) std::cout << "[SUCCESS] - " << log << std::endl;
#define LOG_FAIL(log) std::cout << "[FAILED] - " << log << std::endl;

/*---------------------------------------------------------------------*/
#endif
