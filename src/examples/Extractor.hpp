/*
    Module: Extractor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#ifndef Extractor_hpp
#define Extractor_hpp


#include "FrameManager.hpp"
#include "AIProcessor.hpp"
#include <pthread.h>
#include <signal.h>

#include "../AiEngines/FeatureExtractor/FeatureExtractor.hpp"

class Extractor : public AIProcessor
{
private:
    /* data */
    
    FeatureExtractor* m_extracter;
    FrameManager* m_frameManager = nullptr;
    pthread_mutex_t m_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t m_thread;
    static void* threadFunc(void* args);
    void process();
public:
    Extractor(FrameManager* frameManger);
    ~Extractor();
    void run();
    void stop();
};



#endif
