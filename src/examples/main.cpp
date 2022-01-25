/*
    Module: main.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#include <iostream>
#include <unistd.h>
#include <time.h>
// #include "AIProcessor.hpp"
#include "DecoderThread.hpp"
#include "Traffic.hpp"
#include "Anpr.hpp"
#include "Extractor.hpp"


int main(int argc, char** args){


    std::cout << "=============>Main<==================\n";
    FrameManager* frameManager = new FrameManager();
    AIProcessor* processor = new Traffic(frameManager);
    DecoderThread* decoder = new DecoderThread(frameManager);

    decoder->run();
    usleep(100000);
    processor->run();
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