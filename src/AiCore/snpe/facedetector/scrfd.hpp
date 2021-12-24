#ifndef SCRFDSNPE_HPP
#define SCRFDSNPE_HPP

#include <vector>
#include <memory>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/StringList.hpp"
#include "DlContainer/IDlContainer.hpp"

#define prob_threshold 0.5f
#define _nms_threshold 0.45f
#define input_width 640
#define input_height 640

struct FaceSCRFD
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

class SnpeScrfd
{
    private:
        std::unique_ptr<zdl::SNPE::SNPE> snpeScrfd;
    public:
        SnpeScrfd();
        ~SnpeScrfd();
        int initSnpeScrfd(std::string containerPath, zdl::DlSystem::Runtime_t targetDevice);
        void executeSnpe(std::unique_ptr<zdl::DlSystem::ITensor> &input);
        void executeSnpeScrfd(const cv::Mat &img, std::vector<FaceSCRFD> &face_objects);
        int draw(cv::Mat &rgb, const std::vector<FaceSCRFD> &face_objects);
};

#endif // SCRFDSNPE_HPP