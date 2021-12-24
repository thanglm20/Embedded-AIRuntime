/*
    Module: Object Tracking
    Author: Le Manh Thang
    Created: August 26th, 2021
*/
#ifndef CrosslineDetector_hpp
#define CrosslineDetector_hpp

#include <fstream>
#include <cstdio>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <fstream>
#include "ObjectTracking.hpp"
#include "TypeData.hpp"

class CrosslineDetector
{
private:
    /* data */
    cv::Point beginPoint;
    cv::Point endPoint;
    std::string direction;
    std::vector<std::string> objTracks;
    ObjectTracking* tracking;
    vector<outDataCrossline> listTrack; 
public:
    CrosslineDetector();
    CrosslineDetector(cv::Point beginPoint, cv::Point endPoint, std::string typeCrossing);
    ~CrosslineDetector();
    void setCrossline(cv::Point beginPoint, cv::Point endPoint, std::string typeCrossing);
    void updateCrossline(Mat& frame, vector<ObjectTrace> detected, vector<outDataCrossline>& output);
};



#endif