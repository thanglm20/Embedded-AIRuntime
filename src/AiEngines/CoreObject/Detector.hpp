#ifndef Detector_hpp
#define Detector_hpp

#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <ctime>
#include "Utility.hpp"
#include "../../AiCore/ObjectDetector.hpp"


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

class Detector {
public:
    Detector();
    ~Detector();

    bool detect(const Mat& input, vector<bbox_t>& detected);
private:
    ObjectDetector* objectDetector;
};


#endif /* LocalTrafficEngine_hpp */