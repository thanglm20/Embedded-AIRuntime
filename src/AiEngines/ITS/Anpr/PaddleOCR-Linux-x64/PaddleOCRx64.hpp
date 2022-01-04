/*
    Module: PaddleOCRx64.hpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#ifndef PaddleOCRx64_hpp
#define PaddleOCRx64_hpp

#include "omp.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>


#include "config.h"
#include "ocr_det.h"
#include "ocr_rec.h"
#include "utility.h"
#include <sys/stat.h>

#include "../LicenseOcr.hpp"

using namespace std;
using namespace cv;
using namespace PaddleOCR;

#define CONFIG_FILE "../src/AiEngines/Anpr/PaddleOCR-Linux-x64/config.txt"

class PaddleOCRx64 : public LicenseOcr
{
private:
    int m_width = 0;
    int m_height = 0;
    DBDetector* m_det;
    Classifier* m_cls;
    CRNNRecognizer* m_rec;

public:
    PaddleOCRx64();
    ~PaddleOCRx64 ();
    virtual STATUS init () override;
    virtual std::string recognize (cv::Mat& imgLicense) override;
};

#endif