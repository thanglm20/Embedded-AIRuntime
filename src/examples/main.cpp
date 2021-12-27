/*
    Module: main.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#include <iostream>
#include <unistd.h>
// #include "AIProcessor.hpp"
#include "DecoderThread.hpp"
#include "TrafficDetector.hpp"
#include "Anpr.hpp"
int main(int argc, char** args){


    std::cout << "=============>Main<==================\n";
    FrameManager* frameManager = new FrameManager();
    AIProcessor* processor = new Anpr(frameManager);
    DecoderThread* decoder = new DecoderThread(frameManager);
    
    processor->run();
    decoder->run();
    
    while(1)
    {
        // cv::Mat frame = frameManager->getFrame();
        // if(!frame.empty())
        // {
        //     imshow("frame", frame);
        //     char key = cv::waitKey(1);
        //     if(key == 'q') break;
        // }
        
        sleep(1);
    }
    return 0;
}