/*
    IntrusionDetector.hpp
    Author: ThangLMb
    Date: May 12th, 2021
*/

#ifndef IntrusionDetector_hpp
#define IntrusionDetector_hpp

#include <opencv2/opencv.hpp>
#include "Utility.hpp"
#include <fstream>
#include "TypeData.hpp"
#define NUMBER_FRAME_TRACK = 5;



class IntrusionDetector {

public:
    IntrusionDetector();
    ~IntrusionDetector();


    void setIntrusion(vector<Point> regionsSet, intrusionSet intrusion_set);
    void updateIntrusion(Mat& frame, vector<bbox_t> detected, vector<outDataIntrusion>& output);

private:
    ObjectTracking* tracking;
    vector<Point> regionsSet;
    vector<string> objNames;
    intrusionSet intrusion_set;
    std::vector<outDataIntrusion> listTrack; // list of object in region
    
};

#endif /* OpposeDetector_hpp */
