/*
    Module: OpencvDecoder.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#ifndef OpencvDecoder_hpp
#define OpencvDecoder_hpp



#include "Decoder.hpp"

class OpencvDecoder : public Decoder
{
private:
    /* data */
    cv::VideoCapture m_cap;
public:
    OpencvDecoder();
    ~OpencvDecoder();
    virtual int open( const char* file_video) override;
    virtual cv::Mat getFrame() override;
};
#endif