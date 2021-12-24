/*
    Module: FFmpegDecoder.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#ifndef FFmpegDecoder_hpp
#define FFmpegDecoder_hpp


#include <iostream>
#include <unistd.h>
#include <iostream>
#include <vector>
// FFmpeg
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "Decoder.hpp"

class FFmpegDecoder : public Decoder
{
private:
    /* data */
    AVFrame* decframe = nullptr;
    AVFrame* frame = nullptr;
    AVStream* vstrm = nullptr;
    AVFormatContext* inctx = nullptr;
    SwsContext* swsctx = nullptr;
    AVCodec* vcodec = nullptr;
    int vstrm_idx;
    int dst_width ;
    int dst_height;
    std::vector<uint8_t> framebuf;
    unsigned long nb_frames = 0;
public:
    FFmpegDecoder();
    ~FFmpegDecoder();
    virtual int open( const char* file_video) override;
    virtual cv::Mat getFrame() override;
};




#endif