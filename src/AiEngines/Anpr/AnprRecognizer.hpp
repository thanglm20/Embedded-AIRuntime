/*
    Module: AnprRecognizer.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/

#ifndef AnprRecognizer_hpp
#define AnprRecognizer_hpp

#include <regex>
#include "ocr_db_crnn.hpp"
#include "AnprDetector.hpp"
#include "ObjectTracking.hpp"
#include "MotionDetector.hpp"


#define MAX_COUNT_UNKNOWN 10
#define THRESHOLD_ANPR 0.8
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
    Nations nations;
    MotionDetector motion;
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
    AnprRecognizer(/* args */);
    ~AnprRecognizer();
    int init(Nations nation);
    int recognize( cv::Mat& img, std::vector<PlateInfor>& plates);
    int recognize( cv::Mat& img, std::vector<PlateInfor>& plates, bool checkIsMotion, Rect& rectMotion);
    int trackAnpr( Mat &img, std::vector<PlateInfor>& plates);
};


#endif