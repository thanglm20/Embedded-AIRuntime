/*
    Module: AIProcessor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#ifndef AIProcessor_hpp
#define AIProcessor_hpp

#include "FrameManager.hpp"
#include <pthread.h>
#include <signal.h>
#include <chrono>
class AIProcessor
{
public:
    virtual ~AIProcessor(){;}
    virtual void run() = 0;
    virtual void stop() = 0;
};



#endif
