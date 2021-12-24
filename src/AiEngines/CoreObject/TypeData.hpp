

#ifndef TypeData_hpp
#define TypeData_hpp

#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#define DEBUG_ANDROID 1
#define PATH_LABEL "/system/app/models/snpe/object.txt"

#define NUMBER_FRAME_TRACK = 5;

enum class OBJ_STATUS {OUTSIDE = 0, INSIDE = 1 };

struct intrusionSet
{
    std::vector<std::string> objTracks; 
    int enable_intrusion = 0;
    std::string direction;
};
struct outDataCrossline {
    std::vector<cv::Rect> list_rect;
    cv::Rect rect;
    float score;
    std::string label;
    int track_id;
    int obj_id;
    bool isEvent = false;
};

struct crowdSet
{
    std::vector<std::string> objTracks; 
    int enable_crowd_detection = 0;
    int max_object = 0;
    int timeout = 0;
};

struct disappearanceSet
{
    std::vector<std::string> objTracks; 
    int enable_disappearance = 0;
    int min_object = 0;
    int timeout = 0;
};

struct outDataIntrusion
{
    std::vector<cv::Rect> list_rect;
    cv::Rect rect;
    std::string obj_label;
    float stick_in = 0;
    float stick_out = 0;
    float total_time = 0;
    int obj_id;
    OBJ_STATUS obj_status = OBJ_STATUS::OUTSIDE;
    int track_id;
    int isOutOfFrame = 0;
    bool isEvent = 0;
};


#endif