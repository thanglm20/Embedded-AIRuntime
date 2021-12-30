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
public:
    virtual ~LicenseOcr () {;}
    virtual STATUS init () = 0;
    virtual std::string recognize (cv::Mat& imgLicense) = 0;
};

#endif