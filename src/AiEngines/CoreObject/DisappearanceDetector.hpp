/*
    DisappearanceDetector.hpp
    Author: ThangLMb
    Date: May 12th, 2021
*/

#ifndef DisappearanceDetector_hpp
#define DisappearanceDetector_hpp

#include <opencv2/opencv.hpp>
#include "Utility.hpp"
#include <fstream>
#include "TypeData.hpp"

class DisappearanceDetector {

public:
    DisappearanceDetector();
    ~DisappearanceDetector();
    void setDisappearance(vector<Point> regionsSet, crowdSet crowd_set );
    void updateDisappearance(Mat& frame, vector<bbox_t> detected, vector<outDataIntrusion>& output);
    
private:
    ObjectTracking* tracking;
    vector<Point> regionsSet; 
    vector<string> objNames;
    disappearanceSet disappearance_set;
    std::vector<outDataIntrusion> listTrack;
    
};

#endif /* OpposeDetector_hpp */
