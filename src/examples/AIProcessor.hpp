/*
    Module: AIProcessor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#ifndef AIProcessor_hpp
#define AIProcessor_hpp
#include "../AiEngines/Anpr/AnprRecognizer.hpp"
#include "../AiCore/AIUserFactory.hpp"
#include "FrameManager.hpp"
#include <pthread.h>
#include <signal.h>
class AIProcessor
{
private:
    /* data */
    airuntime::aicore::AIUserFactory* m_executor;
    airuntime::aiengine::AnprRecognizer* m_anpr;
    FrameManager* m_frameManager = nullptr;
    pthread_mutex_t m_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t m_thread;
    static void* threadFunc(void* args);
    void process();
public:
    AIProcessor(FrameManager* frameManger);
    ~AIProcessor();
    void run();
    void stop();
};



#endif
