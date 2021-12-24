
/*
    Module: Oppose
    Author: Le Manh Thang
    Created: Oct 04, 2021
*/
/*
    Algorithm
    Input: Start line, Stop line, Direction, List of allowed object

    Check:
        stop line:  --------->
                        ^
                        .
                        . wrong direction
                        .
        start line: --------->

    Output:

*/
#ifndef Oppose_hpp
#define Oppose_hpp

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

#define NUMBER_LINE_CHECK 10
using namespace std;
using namespace cv;

#define THRES_DETECT_VEHICLE 0.5

struct LinearEquationStartLine{
    double A;
    double B;
    double C;
};

struct VecLine{
    Point tail;
    Point head;
};

struct settingsOppose{
    vector<Point> startLine = vector<Point>(2);
    vector<Point> endLine = vector<Point>(2); 
    vector<VecLine> listLine;
    vector<string> allowedObjects; 
};

struct outDataOppose{
    cv::Rect rect;
    std::string label = "";
    float score;
    int track_id = 0;
    int direction = 0;
    int indexFirstLine = 0;
    bool isOutOfFrame = false;
    bool isTentative = false;
    bool isNewEvent = false;
};

class Oppose
{
private:
    /* data */
    LinearEquationStartLine factorStartLine;
    settingsOppose settings;
    ObjectTracking* tracker = nullptr;
    ObjectDetector* detector = nullptr;
    vector<outDataOppose> listTracked;

public:
    Oppose();
    Oppose(settingsOppose settings);
    ~Oppose();
    int init(settingsOppose settings);
    int set(settingsOppose settings);
    int update(Mat& frame, vector<outDataOppose>& outData);
};
#endif