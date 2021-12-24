/*
    Module: AIProcessor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/


#include "AIProcessor.hpp"



AIProcessor::AIProcessor(FrameManager* frameManger)
{
    this->m_frameManager = frameManger;
}

AIProcessor::~AIProcessor()
{

}

void AIProcessor::run()
{
    pthread_create(&this->m_thread, NULL, threadFunc, this);
}

void AIProcessor::stop()
{
    // pthread_cancel(this->m_thread); // LINUX
    pthread_kill(this->m_thread, SIGUSR1); // ANDROID NDK
    // if(this->m_executor) this->m_executor->release();

}

void* AIProcessor::threadFunc(void* args)
{
    AIProcessor* thread = (AIProcessor*) args;
    thread->process();
    pthread_exit(NULL);
}

void AIProcessor::process()
{
    std::cout << "Creating AI processor\n";
    bool flInit = false;
    this->m_anpr = new airuntime::aiengine::AnprRecognizer(Nations::VN);
    
    VideoWriter writer;          
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');    
    double fps = 5.0;    
    std::string filename = "/data/thanglmb/output/anpr.avi";    
    writer.open(filename, codec, fps, cv::Size(1280, 720));  

    while(1)
    {
        if(!flInit)
        {
            this->m_anpr->init();
            flInit = true;
        }
        pthread_mutex_lock(&this->m_mutex);
        cv::Mat frame0 = this->m_frameManager->getFrame();
        // cv::Mat frame = imread("/data/thanglmb/imgs/19.png");
        cout << "Frame AI process: " << this->m_frameManager->getFrameCounter() << endl;
        cv::Mat frame ;
        frame0.copyTo(frame);
        vector<airuntime::aiengine::PlateInfor> plates;
        if(!frame.empty()) 
        {
            auto start = std::chrono::high_resolution_clock::now();    
            // this->m_anpr->recognize(frame, plates);
            this->m_anpr->trackAnpr(frame, plates);
            auto end = std::chrono::high_resolution_clock::now();    
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            cout << "Performance: AI = " << 1000.0 / duration.count() <<  "FPS, Decoder = " 
                << this->m_frameManager->getFps() << "FPS" << endl;
            cout << "--------------\n";
            for(const auto p : plates)
            {
                cout << p.track_id << ": " << p.license << endl;
                string text = p.track_id + ":" + p.license;
                putText(frame, text, Point(p.rect.x, p.rect.y), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
            }
            resize(frame, frame, Size(1280, 720));
            writer.write(frame);
        }
        pthread_mutex_unlock(&this->m_mutex); 
    }
}