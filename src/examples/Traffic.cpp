/*
    Module: Traffic.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#include "Traffic.hpp"



Traffic::Traffic(FrameManager* frameManger)
{
    this->m_frameManager = frameManger;
}

Traffic::~Traffic()
{

}

void Traffic::run()
{
    pthread_create(&this->m_thread, NULL, threadFunc, this);
}

void Traffic::stop()
{
    // pthread_cancel(this->m_thread); // LINUX
    pthread_kill(this->m_thread, SIGUSR1); // ANDROID NDK
    // if(this->m_executor) this->m_executor->release();

}

void* Traffic::threadFunc(void* args)
{
    Traffic* thread = (Traffic*) args;
    thread->process();
    pthread_exit(NULL);
}

void Traffic::process()
{
    std::cout << "Creating AI processor\n";
    
    // config
    this->m_trafficManager = new airuntime::aiengine::its::TrafficManager();
    while(1)
    {

        pthread_mutex_lock(&this->m_mutex);
        cv::Mat frame0 = this->m_frameManager->getFrame();
        cv::Mat frame ;
        frame0.copyTo(frame);
        
        if(!frame.empty()) 
        {
            auto start = std::chrono::high_resolution_clock::now();    
            // TODO
            std::vector<airuntime::aiengine::its::VehicleTrace> outVehicles; 
            this->m_trafficManager->run(frame, outVehicles);

            auto end = std::chrono::high_resolution_clock::now();    
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            cout << "Performance: AI = " << 1000.0 / duration.count() <<  "FPS, Decoder = " 
                << this->m_frameManager->getFps() << "FPS" << endl;
            cout << "--------------\n";
            
            // post process
            for(auto v : outVehicles)
            {
                rectangle(frame, v.rect, cv::Scalar(255, 255, 255), 2, 8);
                char text[100];
                sprintf(text, "%d", v.track_id);
                putText(frame, text, cv::Point(v.rect.x, v.rect.y), FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 255));
            }

            cv::imshow("traffic", frame);
            cv::waitKey(1);
        }
        
        pthread_mutex_unlock(&this->m_mutex); 
    }
}