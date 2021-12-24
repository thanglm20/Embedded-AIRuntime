/*
    Module: AnprRecognizer.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/

#ifndef AnprRecognizer_hpp
#define AnprRecognizer_hpp

#include <regex>
#include "ocr/ocr_db_crnn.hpp"
#include "AnprDetector.hpp"
#include "../../AiCore/sort-tracker/ObjectTracking.hpp"



#define MAX_COUNT_UNKNOWN 10
#define THRESHOLD_ANPR 0.8
#define THRESHOLD_OCR 0.9
using namespace paddle::lite_api; // NOLINT
using namespace std;
using namespace cv;

typedef struct OcrConfig
{
    std::vector<std::string> dict;
    std::map<std::string, double> configOCR;
    std::shared_ptr<PaddlePredictor> detector;
    std::shared_ptr<PaddlePredictor> recog;
}OcrConfig;

namespace airuntime{
    namespace aiengine{

struct PlateInfor
{
    cv::Mat imgPlate;
    cv::Rect rect;
    std::string license = "";
    std::string typePlate = "";
    float score;
    int track_id = 0;
    bool isOutOfFrame = false;
    bool isNewEvent = false;
    uint8_t countUnknown = 0;
};

class AnprRecognizer
{
    
private:
    /* data */
    Nations m_nations;
    ObjectTracking* tracker = nullptr;
    AnprDetector* detector = nullptr;
    OcrConfig* ocrVN = nullptr;
    OcrConfig* ocrUS = nullptr;
    OcrConfig* ocrMalay = nullptr;
    std::vector<PlateInfor> listPlateTracks;

    std::string readText( cv::Mat& img, Nations nation, float& confidence);
    int initVn(std::string pathDet, std::string pathRecog);
    int initVnSquare(std::string pathDet, std::string pathRecog);
    int initVnRect(std::string pathDet, std::string pathRecog);
    int initUS(std::string pathDet, std::string pathRecog);
    int initMalay(std::string pathDet, std::string pathRecog);
    bool isValidPlate(cv::Mat& img);

public:
    explicit AnprRecognizer(Nations nation);
    ~AnprRecognizer();
    int init();
    int recognize( cv::Mat& img, std::vector<PlateInfor>& plates);
    int trackAnpr( Mat &img, std::vector<PlateInfor>& plates);
};


    }
}



#endif