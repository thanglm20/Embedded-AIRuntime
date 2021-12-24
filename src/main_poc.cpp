#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include "Detector.hpp"
#include "ObjectTracking.hpp"
#include "IntrusionDetector.hpp"
#include "CrosslineDetector.hpp"
#include "CrowdDetector.hpp"


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

Detector* detector = new Detector();
IntrusionDetector* intrusion = new IntrusionDetector();
CrosslineDetector* crossline = new CrosslineDetector();
CrowdDetector* crowd = new CrowdDetector();
int  case_run = -1;
int g_iFrameCounter = 0;
int run(cv::Mat frame)
{
    resize(frame, frame, Size(1280, 720));
    int widthFrame = frame.cols;
    int heightFrame = frame.rows;
    printf("---------------------------------\n");
    // detection
    vector<bbox_t> detected;
    detector->detect(frame, detected);
    //printf("detected: %d\n", detected.size());
    for(auto d : detected)
    {
        rectangle(frame, cv::Rect(d.x, d.y, d.w, d.h), Scalar(255, 0, 0), 1, 8);
    }
    
    switch (case_run)
    {
        case 1:
        {
            // cross line 
            std::vector<std::string> objTracks{"xemay"};
            crossline->setCrossline(cv::Point(0, 200), cv::Point(1280, 200), "in", objTracks);
            vector<outDataCrossline> output_crossline;
            crossline->updateCrossline(frame, detected, output_crossline);
            break;
        }
            
        case 2: 
        {
            // config intrusion 
            intrusionSet setIntrusion;
            setIntrusion.enable_intrusion = 1;
            setIntrusion.direction = "all";
            std::vector<std::string> objTracks{"person"};
            setIntrusion.objTracks = objTracks;
            std::vector<cv::Point> points;
            cv::Point point1(200,200);
            points.push_back(point1);
            cv::Point point2(800, 200);
            points.push_back(point2);
            cv::Point point3(800, 600);
            points.push_back(point3);
            cv::Point point4(200, 600);
            points.push_back(point4);
            //set and run
            intrusion->setIntrusion(points, setIntrusion);
            std::vector<outDataIntrusion> output;
            intrusion->updateIntrusion(frame, detected, output);
            break;
        }
        case 3: 
        {
            // config crowd detection
            crowdSet setCrowd;
            setCrowd.enable_crowd_detection = 1;
            setCrowd.max_object = 2;
            setCrowd.timeout = 1;
            std::vector<std::string> objTracks{"person"};
            setCrowd.objTracks = objTracks;
            std::vector<cv::Point> points_crowd;
            cv::Point point1_crowd(200,200);
            points_crowd.push_back(point1_crowd);
            cv::Point point2_crowd(1000, 200);
            points_crowd.push_back(point2_crowd);
            cv::Point point3_crowd(1000, 600);
            points_crowd.push_back(point3_crowd);
            cv::Point point4_crowd(200, 600);
            points_crowd.push_back(point4_crowd);
            //set and run
            crowd->setCrowd(points_crowd, setCrowd);
            std::vector<outDataIntrusion> output;
            crowd->updateCrowd(frame, detected, output);
            break;
        }
            
        default:
            printf("Please chose options to run\n");
            break;
    }
    cv::imwrite("/data/poc_bca/frame.jpg", frame);
    return 0;
}

int main(int argc, char** argv)
{
    printf("POC BCA \n");

 


    //std::vector<cv::Mat> a = convertVideo2Mat((char*)"/data/poc_bca/crowd.mp4");
    if(argc < 2)
    {
        printf("Please chose [1, 2, 3,...]\n");
        return -1;
    }
    case_run = atoi(argv[1]);

    const char* file_video = "/data/poc_bca/1.mp4";
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
        g_iFrameCounter ++;
        vector<bbox_t> detected;
        auto start =chrono::steady_clock::now();
        detector->detect(image, detected);
        auto end =chrono::steady_clock::now();
	    chrono::duration<double> elapsed = end - start;
        cout<<"Process time: " <<elapsed.count()<<endl;
        //printf("detected: %d\n", detected.size());
        for(auto d : detected)
        {
            rectangle(image, cv::Rect(d.x, d.y, d.w, d.h), Scalar(0, 0, 255), 2, 8);
        }
        
        char pth[100];
        sprintf(pth, "/data/poc_bca/output/%d.jpg", g_iFrameCounter);
        cv::imwrite(pth, image);
        if(g_iFrameCounter > 100)
             break;
        

        //---------------------------------------------------------------------------------------


        }
        std::cout << "Frame of video:" << nb_frames << '\r' << std::flush;  // dump progress
        ++nb_frames;
        next_packet:
        av_free_packet(&pkt);
    } 
    while (!end_of_stream || got_pic);
    std::cout << nb_frames << " frames decoded" << std::endl;

    av_frame_free(&decframe);
    av_frame_free(&frame);
    avcodec_close(vstrm->codec);
    avformat_close_input(&inctx);
    
    
    return 0;
}