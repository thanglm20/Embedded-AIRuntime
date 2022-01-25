/*
    Module: Extractor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#include "Extractor.hpp"



Extractor::Extractor(FrameManager* frameManger)
{
    this->m_frameManager = frameManger;
}

Extractor::~Extractor()
{

}

void Extractor::run()
{
    pthread_create(&this->m_thread, NULL, threadFunc, this);
}

void Extractor::stop()
{
    // pthread_cancel(this->m_thread); // LINUX
    pthread_kill(this->m_thread, SIGUSR1); // ANDROID NDK
    // if(this->m_executor) this->m_executor->release();

}

void* Extractor::threadFunc(void* args)
{
    Extractor* thread = (Extractor*) args;
    thread->process();
    pthread_exit(NULL);
}

void Extractor::process()
{
    std::cout << "Creating AI processor\n";
    
    // config
    this->m_extracter = new FeatureExtractor();
    this->m_extracter->clearData();
    while(1)
    {

        pthread_mutex_lock(&this->m_mutex);
        if(this->m_frameManager->isNewFrame())
        {
            cv::Mat frame0 = this->m_frameManager->getFrame();
            unsigned long frameCounter = this->m_frameManager->getFrameCounter();
            cv::Mat frame ;
            frame0.copyTo(frame);

            if(!frame.empty()) 
            {
                // TODO
                auto start = std::chrono::high_resolution_clock::now();                  
                std::vector<airuntime::aiengine::its::VehicleTrace> outVehicles; 
                this->m_extracter->run(frame, frameCounter);    
                auto end = std::chrono::high_resolution_clock::now();    
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // cout << "Performance: AI = " << 1000.0 / duration.count() <<  "FPS, Decoder = " 
                //     << this->m_frameManager->getFps() << "FPS" << endl;
            }        
        }
        else
        {
            this->m_extracter->saveData();
            std::cout << "Video ended, saved data successfully\n";
            exit(0);
        }
        pthread_mutex_unlock(&this->m_mutex); 
    }
}