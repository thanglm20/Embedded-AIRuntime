/*
    Module: DecoderThread.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#ifndef DecoderThread_hpp
#define DecoderThread_hpp

#include "DecodeFrame.hpp"
#include "FrameManager.hpp"
#include <pthread.h>
#include <signal.h>
class DecoderThread
{
private:
    /* data */
    DecodeFrame* m_decoder;
    FrameManager* m_frameManager = nullptr;
    pthread_mutex_t m_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t m_thread;
    static void* threadFunc(void* args);
    void process();
public:
    DecoderThread(FrameManager* frameManger);
    ~DecoderThread();
    void run();
    void stop();
};



#endif
