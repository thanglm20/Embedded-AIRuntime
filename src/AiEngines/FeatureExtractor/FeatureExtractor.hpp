

/*
    Module: FeatureExtractor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef FeatureExtractor_hpp
#define FeatureExtractor_hpp
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../ITS/TrafficManager.hpp"
#include "BackgroundSubtractor.hpp"
#include "../../nlohmann/json.hpp"

#define PATH_EXTRACTOR "../extractor/"
#define MAX_FRAME_PER_DAY 2592000 // 24 * 3600 * 30fps
using nlohmann::json;

struct ObjInfo
{
    cv::Rect rect;
    unsigned long frame = 0;
    double time = 0.0;
};

struct Features
{
    std::string label = "";
    int ID = 0;
    float score = 0.0;
    unsigned long count = 0;
    std::vector<ObjInfo> info;
    char pathSaving[100];
};

class FeatureExtractor
{
private:
    /* data */
    airuntime::aiengine::its::TrafficManager* m_trafficManager;
    BackgroundSub* m_subtractor;
    unsigned long m_nFrame = 0;
    double m_fTime = 0.0;
    double m_stickStart = 0;
    std::vector<Features> m_listVehicles;
    
public:
    FeatureExtractor(/* args */);
    ~FeatureExtractor();
    void run(cv::Mat& img, unsigned long frameCounter);
    cv::Mat getBackground();
    void clearData();
    void saveData();
};


#endif