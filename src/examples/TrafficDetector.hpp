/*
    Module: TrafficDetector.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/

#ifndef TrafficDetector_hpp
#define TrafficDetector_hpp

#include <pthread.h>
#include <signal.h>
#include "FrameManager.hpp"
#include "AIProcessor.hpp"
#include "../AiCore/AIUserFactory.hpp"

class TrafficDetector : public AIProcessor
{
private:
    /* data */
    airuntime::aicore::AIUserFactory* m_executor;
    FrameManager* m_frameManager = nullptr;
    pthread_mutex_t m_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t m_thread;
    static void* threadFunc(void* args);
    void process();
public:
    TrafficDetector(FrameManager* frameManger);
    ~TrafficDetector();
    virtual void run() override;
    virtual void stop() override;
};

#endif