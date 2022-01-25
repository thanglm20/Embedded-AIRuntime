/*
    Module: Violation.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#ifndef Violation_hpp
#define Violation_hpp

#include <iostream>
#include "../../../AiCore/AITypeData.hpp"
#include "../../../AiCore/sort-tracker/ObjectTracking.hpp"

using namespace std;

///////////////////////////////////////////
// oppose settings
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
    bool isUsed = false;
    vector<Point> startLine = vector<Point>(2);
    vector<Point> endLine = vector<Point>(2); 
    vector<VecLine> listLine;
    vector<string> allowedObjects; 
};
///////////////////////////////////////////
// invade settings
struct settingsInvade{
    bool isUsed = false;
};

struct ViolationSettings
{
    settingsOppose oppose;
    settingsInvade invade;
};

class Violation 
{
    ViolationSettings m_settings;
public:
    explicit Violation(ViolationSettings settings) : m_settings(settings){}
    virtual ~Violation () {;}
    virtual STATUS init () = 0;
    virtual STATUS process (const cv::Mat& frame, ObjectTracking* tracker, 
                            std::vector<TrackingTrace>& tracks
                            ) = 0;
    ViolationSettings getSettings() {return this->m_settings;}
};

#endif