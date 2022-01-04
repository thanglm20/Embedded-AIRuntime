/*
    Module: AnprRecognizer.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/

#ifndef AnprRecognizer_hpp
#define AnprRecognizer_hpp

#include <regex>
#include "AnprDetector.hpp"
#include "AnprConfig.hpp"
#include "LicenseOcr.hpp"
#include "../../../AiCore/sort-tracker/ObjectTracking.hpp"




using namespace std;
using namespace cv;


namespace airuntime{
    namespace aiengine{



class AnprRecognizer
{
    
private:
    /* data */
    Nations m_nations;
    ObjectTracking* tracker = nullptr;
    AnprDetector* detector = nullptr;
    LicenseOcr* m_licenseOcr;
    std::vector<PlateInfor> listPlateTracks;

    bool isValidPlate(cv::Mat& img);

public:
    explicit AnprRecognizer(Nations nation);
    ~AnprRecognizer();
    int init();
    int recognize( cv::Mat& img, std::vector<PlateInfor>& plates);
    int trackAnpr( Mat &img, std::vector<PlateInfor>& plates);
};


    }
}



#endif