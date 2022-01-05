/*
    Module: Traffic.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#ifndef Traffic_hpp
#define Traffic_hpp


#include "FrameManager.hpp"
#include "AIProcessor.hpp"
#include <pthread.h>
#include <signal.h>

#include "../AiEngines/ITS/TrafficManager.hpp"

class Traffic : public AIProcessor
{
private:
    /* data */
    
    airuntime::aiengine::its::TrafficManager * m_trafficManager;
    FrameManager* m_frameManager = nullptr;
    pthread_mutex_t m_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t m_thread;
    static void* threadFunc(void* args);
    void process();
public:
    Traffic(FrameManager* frameManger);
    ~Traffic();
    void run();
    void stop();
};



#endif
