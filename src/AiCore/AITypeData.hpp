
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

using namespace std;
using namespace cv;


// #define ANDROID

#define USE_SNPE
// #define USE_NCNN
#define USE_TFLITE
#define USE_MNN

namespace airuntime{
    enum class ExecutorType {SNPE = 0, NCNN = 1, TFLITE = 2, MNN = 3};
    enum class DeviceType {CPU = 0, GPU = 1, DSP = 2};
    enum class AlgTypeAI {DETECT = 0, CLASSIFY = 1};
    
}
typedef struct ObjectTrace
{
    cv::Rect2f rect; // xmin, ymin, width, height
    std::string label;
    float score;
    int obj_id;   
}ObjectTrace;

enum STATUS {
        FAIL = -1,
        SUCCESS = 0,
        UNSUPPORTED = 2,
        INVALID_ARGS = 3,
    };
#define SNPE_LIB_PATH "/data/snpe/dsp/lib"

#define PATH_DICTIONARY "/data/thanglmb/models/paddle/anpr.txt"

#define DIR_OCR_DET_VN "/data/thanglmb/models/paddle/ocr_detect_vn.nb" 
#define DIR_OCR_RECOG_VN "/data/thanglmb/models/paddle/ocr_recog_vn.nb"

#define DIR_OCR_DET_US "/data/thanglmb/models/paddle/us-det.nb" 
#define DIR_OCR_RECOG_US "/data/thanglmb/models/paddle/us-rec.nb"

#define DIR_OCR_DET_MALAY "/data/thanglmb/models/malay-det.nb"
#define DIR_OCR_RECOG_MALAY "/data/thanglmb/models/malay-rec.nb"

#define DIR_MODEL_OBJECTDETECION_NCNN "/data/thanglmb/models/ncnn/" // G1
#define DIR_LABEL_OBJECTDETECION_NCNN "/data/thanglmb/models/ncnn/" // G1

#define DIR_MODEL_OBJECTDETECION_SNPE "/data/thanglmb/models/snpe/" // G2
#define DIR_LABEL_OBJECTDETECION_SNPE "/data/thanglmb/models/snpe/" // G2

#define DIR_MODEL_OBJECTCLASSIFICATION "/system/app/ai_config/models/snpe/" //G2
#define DIR_LABEL_OBJECTCLASSIFICATION "/system/app/ai_config/models/snpe/" //G2

#define DIR_MODEL_OBJECTCLASSIFICATION_TFLITE "/system/app/ai_config/models/tflite/multilabel.tflite"
#define DIR_LABEL_OBJECTDETECION_TFLITE "/system/app/ai_config/models/tflite/" // G2
#define DIR_MODEL_OBJECTDETECION_TFLITE "/system/app/ai_config/models/tflite/" 

#define DIR_MODEL_OBJECTDETECION_MNN "/system/app/ai_config/models/mnn/"
#define DIR_LABEL_OBJECTDETECION_MNN "/system/app/ai_config/models/mnn/"

#define LOG_INFO(log) std::cout << "[INFO] - " << log << std::endl;
#define LOG_SUCCESS(log) std::cout << "[SUCCESS] - " << log << std::endl;
#define LOG_FAIL(log) std::cout << "[FAILED] - " << log << std::endl;

/*---------------------------------------------------------------------*/
#endif
