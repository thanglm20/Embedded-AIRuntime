/*
    Module: DecoderThread.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifdef USE_FFMPEG
#include "FFmpegDecoder.hpp"
#else
#include "OpencvDecoder.hpp"
#endif

#include "DecoderThread.hpp"
#include <chrono>

DecoderThread::DecoderThread(FrameManager* frameManger)
{
    this->m_frameManager = frameManger;
}

DecoderThread::~DecoderThread()
{
}

void DecoderThread::run()
{
    pthread_create(&this->m_thread, NULL, threadFunc, this);
}

void DecoderThread::stop()
{

    // pthread_cancel(this->m_thread); // LINUX
    pthread_kill(this->m_thread, SIGUSR1); // ANDROID NDK
    if(this->m_decoder) delete this->m_decoder;
}
 
void* DecoderThread::threadFunc(void* args)
{
    DecoderThread* thread = (DecoderThread*) args;
    thread->process();
    pthread_exit(NULL);
}

void DecoderThread::process()
{
    std::cout << "Creating decoder thread\n";

    #ifdef USE_FFMPEG
        this->m_decoder = new FFmpegDecoder();
    #else 
        this->m_decoder = new OpencvDecoder();
    #endif

    const char* video_file = (char*)"/home/thanglmb/Downloads/abc.mp4";
    int ret = this->m_decoder->open(video_file);

    if(ret == 0)
    while(1)
    {     
        auto start = std::chrono::high_resolution_clock::now();    
        cv::Mat frame = this->m_decoder->getFrame();
        auto end = std::chrono::high_resolution_clock::now();    
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        float fps = 1000.0 / duration.count();
        if(!frame.empty())
        {
            pthread_mutex_lock(&this->m_mutex);
            this->m_frameManager->updateFrame(frame, fps);
            pthread_mutex_unlock(&this->m_mutex);
            std::cout << "Decoded frame: " << frame.cols  << "x" << frame.rows
                << ", frame " << this->m_frameManager->getFrameCounter() << std::endl;
        }
        else 
        {
            std::cout << "Frame is empty\n";
            break;
        }
    }        
    else 
        std::cout << "Open video file error\n";
  
}
