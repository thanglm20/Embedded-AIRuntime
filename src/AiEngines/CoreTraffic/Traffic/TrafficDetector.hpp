#ifndef TrafficDetector_hpp
#define TrafficDetector_hpp

#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <ctime>
#include "Utility.hpp"



struct OutputData {
    Rect box;
    float score;
    string label;
    int id;
    int obj_id;
    vector<Rect> brect;
    bool isDisappear;
    bool isKill;
    bool isNew;
};
struct Track {
    int id;
    string label;
    vector<Rect> brect;
    int count;
    int64 tick;
    bool outOfTheFrame;
};

class TrafficDetector {
public:
    TrafficDetector();
    ~TrafficDetector();

    bool detect(const Mat& input, vector<bbox_t>& detected);
private:
    ObjectDetector* detector;
};

#endif /* LocalTrafficEngine_hpp */