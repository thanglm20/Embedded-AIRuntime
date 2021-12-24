/*
    CrowdDetector.hpp
    Author: ThangLMb
    Date: May 12th, 2021
*/

#ifndef CrowdDetector_hpp
#define CrowdDetector_hpp

#include <opencv2/opencv.hpp>
#include "Utility.hpp"
#include <fstream>
#include "TypeData.hpp"

class CrowdDetector {

public:
    CrowdDetector();
    ~CrowdDetector();
    void setCrowd(vector<Point> regionsSet, crowdSet crowd_set );
    void updateCrowd(Mat& frame, vector<bbox_t> detected, vector<outDataIntrusion>& output);
    
private:
    ObjectTracking* tracking;
    vector<Point> regionsSet; 
    vector<string> objNames;
    crowdSet crowd_set;
    std::vector<outDataIntrusion> listTrack;
    
};

#endif /* OpposeDetector_hpp */
