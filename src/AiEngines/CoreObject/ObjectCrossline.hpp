/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: ObjectInvade.cpp
 Author: LE MANH THANG
 Created: May 10th,2021
 Description: 
********************************************************************************/
#ifndef ObjectCrossline_hpp
#define ObjectCrossline_hpp

#include <fstream>
#include <cstdio>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <fstream>
#include "Utility.hpp"

#define DEBUG_ANDROID 1

struct outDataCrossline {
    Rect rect;
    float score;
    string label;
    int track_id;
    int obj_id;
    vector<Rect> list_rect;
    bool isEvent = false;
};

class ObjectCrossline
{
private:
    /* data */
    ObjectDetector* objectDetector;  
    cv::Point beginPoint;
    cv::Point endPoint;
    std::string direction;

    ObjectTracking* tracking;
    vector<outDataCrossline> listTrack; 

    vector<string> objNames;
    bool checkTraces(Trace traces, int width, int height, int oppType); 
public:
    //std::string nameObject, cv::Point beginPoint, cv::Point endPoint, int direction
    ObjectCrossline(cv::Point beginPoint, cv::Point endPoint, std::string typeCrossing, std::string nameObject);
    ~ObjectCrossline();
    void setData(cv::Point beginPoint, cv::Point endPoint, std::string typeCrossing, std::string nameObject);
    void updateCrossline(Mat& frame, vector<bbox_t> detected, vector<outDataCrossline>& output);
};



#endif