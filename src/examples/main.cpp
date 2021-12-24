/*
    Module: main.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#include <iostream>

#include "AIProcessor.hpp"
#include "DecoderThread.hpp"

int main(int argc, char** args){


    std::cout << "=============>Main<==================\n";
    FrameManager* frameManager = new FrameManager();
    AIProcessor* processor = new AIProcessor(frameManager);
    DecoderThread* decoder = new DecoderThread(frameManager);
    
    processor->run();
    decoder->run();
    
    while(1)
    {
        sleep(1);
    }
    return 0;
}