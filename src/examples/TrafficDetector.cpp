

#include "TrafficDetector.hpp"


TrafficDetector::TrafficDetector(FrameManager* frameManger)
{
    this->m_frameManager = frameManger;
}
TrafficDetector::~TrafficDetector()
{

}
void TrafficDetector::run()
{
    pthread_create(&this->m_thread, NULL, threadFunc, this);
}
void TrafficDetector::stop()
{
    // pthread_cancel(this->m_thread); // LINUX
    pthread_kill(this->m_thread, SIGUSR1); // ANDROID NDK
    // if(this->m_executor) this->m_executor->release();
}

void* TrafficDetector::threadFunc(void* args)
{
    TrafficDetector* thread = (TrafficDetector*) args;
    thread->process();
    pthread_exit(NULL);
}

void TrafficDetector::process()
{
    std::cout << "Creating AI processor\n";
    bool flInit = false;
    // this->m_executor = new airuntime::aicore::AIUserFactory(airuntime::ExecutorType::NCNN,
    //                                                         airuntime::DeviceType::CPU,
    //                                                         airuntime::AlgTypeAI::DETECT,
    //                                                         "../models/traffic.txt",
    //                                                         "../models/traffic.dlc");
    this->m_executor = new airuntime::aicore::AIUserFactory(airuntime::ExecutorType::NCNN,
                                                            airuntime::DeviceType::CPU,
                                                            airuntime::AlgTypeAI::DETECT,
                                                            "../models/traffic.txt",
                                                            "../models/traffic.bin",
                                                            "../models/traffic.param");
    while(1)
    {
        pthread_mutex_lock(&this->m_mutex);
        cv::Mat frame0 = this->m_frameManager->getFrame();
        cv::Mat frame;
        frame0.copyTo(frame);
        if(!frame.empty()) 
        {
            
            vector<ObjectTrace> objects;
            auto start = std::chrono::high_resolution_clock::now();    
            this->m_executor->run(frame, objects, 0.5);
            auto end = std::chrono::high_resolution_clock::now();    
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            cout << "Performance: AI = " << 1000.0 / duration.count() <<  "FPS, Decoder = " 
                << this->m_frameManager->getFps() << "FPS" << endl;

            for(const auto p : objects)
            {
                rectangle(frame, p.rect, Scalar(255, 255, 255), 2, 8);
            }
            imshow("traffic", frame);
            waitKey(1);
        }
        pthread_mutex_unlock(&this->m_mutex); 
    }
}