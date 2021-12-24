#ifndef WrongLaneDetector_hpp
#define WrongLaneDetector_hpp

#include <opencv2/opencv.hpp>
#include "TrafficDetector.hpp"
#include "Utility.hpp"

class WrongLaneDetector {
public:
    WrongLaneDetector();

    void update(Mat& frame, vector<bbox_t> detected, vector<OutputData>& output);

    void reset();
    void setData(vector<vector<Point>> regionsSet, vector<string> labels);
private:
    ObjectTracking* tracking;

    vector<string> objNames;
    vector<vector<Point>> regionsSet;
    vector<string> labelSet;
};

#endif /* WrongLaneDetector_hpp */
