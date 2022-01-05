

#include "BackgroundSubtractor.hpp"


BackgroundSub::BackgroundSub(/* args */){
    this->m_subMOG2 = cv::createBackgroundSubtractorMOG2(40, 300, false);
}

BackgroundSub::~BackgroundSub(){}

void BackgroundSub::run(cv::Mat& img){
    this->m_subMOG2->apply(img, this->m_mask);
}

void BackgroundSub::extract(cv::Mat& img)
{
    this->m_subMOG2->apply(img, this->m_mask);
}
cv::Mat BackgroundSub::getBackground()
{
    cv::Mat background;
    this->m_subMOG2->getBackgroundImage(background);
        return background;
}
void BackgroundSub::release(){
    this->m_subMOG2.release();
}