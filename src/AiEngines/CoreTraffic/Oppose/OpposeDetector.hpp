#ifndef OpposeDetector_hpp
#define OpposeDetector_hpp

#include <opencv2/opencv.hpp>
#include "TrafficDetector.hpp"
#include "Utility.hpp"
#include <fstream>


class OpposeDetector {
public:
    OpposeDetector();
    ~OpposeDetector();

    void update(Mat& frame, vector<bbox_t> detected, vector<OutputData>& output);
    void setData(vector<vector<Point>> regionsSet, vector<string> labels);
private:
    ObjectTracking* tracking;
    vector<vector<Point>> regionsSet;
    vector<string> labelSet;
    int direction;
    vector<Track> listTrack; 

    vector<string> objNames;
    bool checkTraces(Trace traces, int width, int height, int oppType);  
};

#endif /* OpposeDetector_hpp */
