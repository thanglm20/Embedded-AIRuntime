/*
    Module: LicenseOcr.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#ifndef LicenseOcr_hpp
#define LicenseOcr_hpp

#include <iostream>
#include "../../AiCore/AITypeData.hpp"
#include "AnprConfig.hpp"

using namespace std;

class LicenseOcr
{
private:
    int m_width = 0;
    int m_height = 0;
public:
    virtual ~LicenseOcr () {;}
    virtual STATUS init () = 0;
    virtual std::string recognize (cv::Mat& imgLicense) = 0;
    int getWidth() {return m_width;}
    int getHeight() {return m_height;}
};

#endif