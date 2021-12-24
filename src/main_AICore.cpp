/*
    Module: AnprRecognizer.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#include <opencv2/opencv.hpp>
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
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "AnprRecognizer.hpp"
#include "Detector.hpp"
#include "ObjectTracking.hpp"
#include "IntrusionDetector.hpp"
#include "CrosslineDetector.hpp"
#include "CrowdDetector.hpp"
#include "AnprYoloText.hpp"

#include "SnpeRetinanet.hpp"

// Detector* detector = new Detector();
// IntrusionDetector* intrusion = new IntrusionDetector();
// CrosslineDetector* crossline = new CrosslineDetector();
// CrowdDetector* crowd = new CrowdDetector();
// 
// int  case_run = -1;
// int g_iFrameCounter = 0;
// int run(cv::Mat frame)
// {
//     resize(frame, frame, Size(1280, 720));
//     int widthFrame = frame.cols;
//     int heightFrame = frame.rows;
//     printf("---------------------------------\n");
//     // detection
//     vector<bbox_t> detected;
//     detector->detect(frame, detected);
//     //printf("detected: %d\n", detected.size());
//     for(auto d : detected)
//     {
//         rectangle(frame, cv::Rect(d.x, d.y, d.w, d.h), Scalar(255, 0, 0), 1, 8);
//     }
    
//     switch (case_run)
//     {
//         case 1:
//         {
//             // cross line 
//             std::vector<std::string> objTracks{"xemay"};
//             crossline->setCrossline(cv::Point(0, 200), cv::Point(1280, 200), "in", objTracks);
//             vector<outDataCrossline> output_crossline;
//             crossline->updateCrossline(frame, detected, output_crossline);
//             break;
//         }
            
//         case 2: 
//         {
//             // config intrusion 
//             intrusionSet setIntrusion;
//             setIntrusion.enable_intrusion = 1;
//             setIntrusion.direction = "all";
//             std::vector<std::string> objTracks{"person"};
//             setIntrusion.objTracks = objTracks;
//             std::vector<cv::Point> points;
//             cv::Point point1(200,200);
//             points.push_back(point1);
//             cv::Point point2(800, 200);
//             points.push_back(point2);
//             cv::Point point3(800, 600);
//             points.push_back(point3);
//             cv::Point point4(200, 600);
//             points.push_back(point4);
//             //set and run
//             intrusion->setIntrusion(points, setIntrusion);
//             std::vector<outDataIntrusion> output;
//             intrusion->updateIntrusion(frame, detected, output);
//             break;
//         }
//         case 3: 
//         {
//             // config crowd detection
//             crowdSet setCrowd;
//             setCrowd.enable_crowd_detection = 1;
//             setCrowd.max_object = 2;
//             setCrowd.timeout = 1;
//             std::vector<std::string> objTracks{"person"};
//             setCrowd.objTracks = objTracks;
//             std::vector<cv::Point> points_crowd;
//             cv::Point point1_crowd(200,200);
//             points_crowd.push_back(point1_crowd);
//             cv::Point point2_crowd(1000, 200);
//             points_crowd.push_back(point2_crowd);
//             cv::Point point3_crowd(1000, 600);
//             points_crowd.push_back(point3_crowd);
//             cv::Point point4_crowd(200, 600);
//             points_crowd.push_back(point4_crowd);
//             //set and run
//             crowd->setCrowd(points_crowd, setCrowd);
//             std::vector<outDataIntrusion> output;
//             crowd->updateCrowd(frame, detected, output);
//             break;
//         }
            
//         default:
//             printf("Please chose options to run\n");
//             break;
//     }
//     cv::imwrite("/data/test/poc_bca/frame.jpg", frame);
//     return 0;
// }


int main(int argc, char** argv)
{   
    // system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SNPE_ROOT/lib");
    // system("export PATH=$PATH:/system/bin");
    // system("export ADSP_LIBRARY_PATH=\"$SNPE_ROOT/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp\"");
    // system("export ADSP_LIBRARY_PATH=\"$SNPE_ROOT/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp\"");
    // system("echo [INFO] - Export env for DSP successfully\n");
    int numberFrame = 1;
    if(argc > 1)    numberFrame = atoi(argv[1]);
    char out_path[100];
    char in_path[100];
    if(argc > 2)
    {
        sprintf(in_path, "/data/thanglmb/imgs/%s", argv[2]);
        sprintf(out_path, "/data/thanglmb/out/%s", argv[2]);
    }
    system("rm /data/thanglmb/out/*");
    AnprRecognizer* anpr = new AnprRecognizer();
    anpr->init();
    printf("Init ANPR successfully\n");
    // const char* file_video = "/data/thanglmb/videos/short.mp4";
    const char* file_video = "/data/thanglmb/videos/road.mp4";
    VideoWriter writer;    
    //int codec = VideoWriter::fourcc('M', 'P', '4', 'V');       
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');    
    double fps = 3.0;    
    std::string filename = "/data/thanglmb/out/out.avi";    
    writer.open(filename, codec, fps, cv::Size(1280,720));   
    // initialize FFmpeg library
    av_register_all();
//  av_log_set_level(AV_LOG_DEBUG);
    int ret;

    // open input file context
    AVFormatContext* inctx = nullptr;
    ret = avformat_open_input(&inctx, file_video, nullptr, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avforamt_open_input(\"" << file_video << "\"): ret=" << ret;
        return 2;
    }
    // retrive input stream information
    ret = avformat_find_stream_info(inctx, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avformat_find_stream_info: ret=" << ret;
        return 2;
    }

    // find primary video stream
    AVCodec* vcodec = nullptr;
    ret = av_find_best_stream(inctx, AVMEDIA_TYPE_VIDEO, -1, -1, &vcodec, 0);
    if (ret < 0) {
        std::cerr << "fail to av_find_best_stream: ret=" << ret;
        return 2;
    }
    const int vstrm_idx = ret;
    AVStream* vstrm = inctx->streams[vstrm_idx];

    // open video decoder context
    ret = avcodec_open2(vstrm->codec, vcodec, nullptr);
    if (ret < 0) {
        std::cerr << "fail to avcodec_open2: ret=" << ret;
        return 2;
    }

    // print input video stream informataion
    std::cout
        << "file_video: " << file_video << "\n"
        << "format: " << inctx->iformat->name << "\n"
        << "vcodec: " << vcodec->name << "\n"
        << "size:   " << vstrm->codec->width << 'x' << vstrm->codec->height << "\n"
        << "fps:    " << av_q2d(vstrm->codec->framerate) << " [fps]\n"
        << "length: " << av_rescale_q(vstrm->duration, vstrm->time_base, {1,1000}) / 1000. << " [sec]\n"
        << "pixfmt: " << av_get_pix_fmt_name(vstrm->codec->pix_fmt) << "\n"
        << "frame:  " << vstrm->nb_frames << "\n"
        << std::flush;

    // initialize sample scaler
    const int dst_width = vstrm->codec->width;
    const int dst_height = vstrm->codec->height;
    const AVPixelFormat dst_pix_fmt = AV_PIX_FMT_BGR24;
    SwsContext* swsctx = sws_getCachedContext(
        nullptr, vstrm->codec->width, vstrm->codec->height, vstrm->codec->pix_fmt,
        dst_width, dst_height, dst_pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsctx) {
        std::cerr << "fail to sws_getCachedContext";
        return 2;
    }
    std::cout << "output: " << dst_width << 'x' << dst_height << ',' << av_get_pix_fmt_name(dst_pix_fmt) << std::endl;

    // allocate frame buffer for output
    AVFrame* frame = av_frame_alloc();
    std::vector<uint8_t> framebuf(avpicture_get_size(dst_pix_fmt, dst_width, dst_height));
    avpicture_fill(reinterpret_cast<AVPicture*>(frame), framebuf.data(), dst_pix_fmt, dst_width, dst_height);

    // decoding loop
    AVFrame* decframe = av_frame_alloc();
    unsigned nb_frames = 0;
    unsigned frameProcessed = 0;
    bool end_of_stream = false;
    int got_pic = 0;
    AVPacket pkt;
    do {
        if (!end_of_stream) {
            // read packet from input file
            ret = av_read_frame(inctx, &pkt);
            if (ret < 0 && ret != AVERROR_EOF) {
                std::cerr << "fail to av_read_frame: ret=" << ret;
                //return 2;
            }
            if (ret == 0 && pkt.stream_index != vstrm_idx)
                goto next_packet;
            end_of_stream = (ret == AVERROR_EOF);
        }
        if (end_of_stream) {
            // null packet for bumping process
            av_init_packet(&pkt);
            pkt.data = nullptr;
            pkt.size = 0;
        }
        // decode video frame
        avcodec_decode_video2(vstrm->codec, decframe, &got_pic, &pkt);
        if (!got_pic)
            goto next_packet;
        // convert frame to OpenCV matrix
        sws_scale(swsctx, decframe->data, decframe->linesize, 0, decframe->height, frame->data, frame->linesize);
        {
        cv::Mat image(dst_height, dst_width, CV_8UC3, framebuf.data(), frame->linesize[0]);
        // TO DO with image
        //---------------------------------------------------------------------------------------
        // detection
        if(nb_frames%10 == 0)
        if(!image.empty())
        {
            frameProcessed++;
            if(argc > 2) 
            {
                std::cout << "Input image: " << in_path << std::endl;
                image = cv::imread(in_path);
            }
            cv::Mat srcimg;
            cv::resize(image, srcimg, cv::Size(1280, 720));
            
            auto start = std::chrono::system_clock::now();

            std::vector<PlateInfor> plates;
            anpr->recognize(srcimg, plates);

            auto end = std::chrono::system_clock::now();    
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            printf("------------------<frame - %d>----------------------\n", nb_frames);
            std::cout << "Time: "
                    << double(duration.count()) *
                        std::chrono::microseconds::period::num /
                        std::chrono::microseconds::period::den
                    << "s" << std::endl;
            printf("Count number plate: %d\n", plates.size());
            for(int i = 0; i < plates.size(); i++)
            {
                printf("-------------\n");
                std::cout << "License: " << plates[i].license << std::endl;
                std::cout << "Type: " << plates[i].typePlate << std::endl;
                std::cout << "Score: " << plates[i].score << std::endl;
                cv::rectangle(srcimg, plates[i].rect, cv::Scalar(255, 0, 0), 1, 8);
                char text_draw[100];
                sprintf(text_draw, "%s-%0.2f", plates[i].license.c_str(), plates[i].score);
                cv::putText(srcimg, text_draw, cv::Point(plates[i].rect.x, plates[i].rect.y), cv::FONT_HERSHEY_SIMPLEX, srcimg.rows * 0.001, cv::Scalar(0, 255, 0), 1);

                
                char w_h[100];
                // sprintf(w_h, "w:%d-h:%d", plates[i].rect.width, plates[i].rect.height);
                // sprintf(w_h, "var:%d", plates[i].variance);
                // cv::putText(srcimg, w_h, cv::Point(plates[i].rect.x , plates[i].rect.y + plates[i].rect.height), cv::FONT_HERSHEY_SIMPLEX, srcimg.rows * 0.001, cv::Scalar(0, 255, 0), 1);
                
                // char path_license[100];
                // sprintf(path_license, "/data/thanglmb/out/%s.jpg",  plates[i].license.c_str());
                // cv::imwrite(path_license, plates[i].imgPlate);
                
                // char path_license_ori[100];
                // sprintf(path_license_ori, "/data/thanglmb/out/%s_ori.jpg",  plates[i].license.c_str());
                // cv::imwrite(path_license_ori, plates[i].ori_img);
            }
            
            char path[100];
            //sprintf(path, "/data/thanglmb/out/%d.jpg", frameProcessed);
            if(argc > 2) 
            {
                sprintf(path, "%s", out_path);
                std::cout << "Image out: " << path << std::endl;
                cv::imwrite(path, srcimg);
            }
            
            writer.write(srcimg);
        
            if(frameProcessed >= numberFrame)
                break;
        }
        }
        //std::cout << "Frame of video:" << nb_frames << '\r' << std::flush;  // dump progress
        ++nb_frames;
        next_packet:
        av_free_packet(&pkt);
    } 
    while (!end_of_stream || got_pic);
    std::cout << nb_frames << " frames decoded" << std::endl;
    writer.release();   
    av_frame_free(&decframe);
    av_frame_free(&frame);
    avcodec_close(vstrm->codec);
    avformat_close_input(&inctx);
    
    
    return 0;
}


