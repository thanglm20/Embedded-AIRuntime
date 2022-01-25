
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
#include "Violation.hpp"

#define NUMBER_LINE_CHECK 10



// struct LinearEquationStartLine{
//     double A;
//     double B;
//     double C;
// };

// struct VecLine{
//     Point tail;
//     Point head;
// };

// struct settingsOppose{
//     vector<Point> startLine = vector<Point>(2);
//     vector<Point> endLine = vector<Point>(2); 
//     vector<VecLine> listLine;
//     vector<string> allowedObjects; 
// };

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

class Oppose : public Violation
{
private:
    /* data */
    LinearEquationStartLine m_factorStartLine;
    settingsOppose m_settings;
    vector<outDataOppose> m_listTracked;

public:

    explicit Oppose(ViolationSettings settings);
    ~Oppose();
    STATUS init () override;
    STATUS process (const cv::Mat& frame, ObjectTracking* tracker, std::vector<TrackingTrace>& tracks) override;
};
#endif