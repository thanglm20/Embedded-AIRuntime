

#ifndef Intrusion_hpp
#define Intrusion_hpp

//==========================================================
// New accessing:
    // object counter > n => timeCounter++: timeCounter > time >>> New Event, update time

// New go outside:
    // object counter < n => reset timeCounter 

// timeCounter > time, object counter > n >>> New Event

//==========================================================


#include <fstream>
#include <cstdio>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <fstream>
#include <math.h>
#include "ObjectTracking.hpp"
#include "ObjectDetector.hpp"
#include "ColorClassifier.hpp"

using namespace std;
using namespace cv;
#define THRES_DETECT_VEHICLE 0.5
#define MAX_EVENT_COUNTER 10
struct settingsIntrusion{
    vector<string> arListObjects;
    vector<Point> arRegionsSet; 
    uint8_t cTimeOut = 0;
    uint8_t cTimeRepeat = 0;
    uint8_t cObjectCounter = 0;
};

struct outDataIntrusion{
    cv::Rect rect;
    std::string label = "";
    std::string color;
    double cTimeCounter = 0;
    double fTimeAccessed = 0;
    float score;
    int track_id = 0;
    bool isOutOfFrame = false;
};


class Intrusion
{
private:
    /* data */
    bool m_flFirstEvent = false;
    bool m_flIsReadyEvent = true;
    bool m_flIsNewAccessing = false;
    bool m_flIsNewOutSide = false;
    double m_fTimeStartCounter = 0.0;
    uint16_t m_cTimeCounter = 0;
    uint16_t m_cRepeatCounter = 1;

    settingsIntrusion m_settings;
    ObjectTracking* m_tracker = nullptr;
    ObjectDetector* m_detector = nullptr;
    vector<outDataIntrusion> m_listTracked;
    ColorClassifier m_ColorClassifier;
public:
    
    Intrusion();
    Intrusion(settingsIntrusion settings);
    ~Intrusion();
    int init(settingsIntrusion settings);
    int set(settingsIntrusion settings);
    int update(Mat& frame, vector<outDataIntrusion>& outData);
};

#endif