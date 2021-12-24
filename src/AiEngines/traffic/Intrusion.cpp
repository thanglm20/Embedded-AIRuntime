/*
    Module: Intrusion
    Author: Le Manh Thang
    Created: Oct 04, 2021
*/


#include "Intrusion.hpp"

Intrusion::Intrusion()
{

}
Intrusion::Intrusion(settingsIntrusion settings)
{

    this->m_detector = new ObjectDetector();
    this->m_tracker = new ObjectTracking();
    this->m_settings = settings;
    
    if(this->m_detector->initObjectDetector("ncnn", "traffic", "CPU") != STATUS_SUCCESS)
    // if(this->m_detector->initObjectDetector("snpe", "traffic", "CPU") != STATUS_SUCCESS)
    {
        LOG_FAIL("Init detector failed");            
    }
}

Intrusion::~Intrusion()
{
    if(this->m_detector) delete this->m_detector;
    if(this->m_tracker) delete this->m_tracker;
}

int Intrusion::init(settingsIntrusion settings)
{
    this->m_detector = new ObjectDetector();
    this->m_tracker = new ObjectTracking();
    this->m_settings = settings;
    if(this->m_detector->initObjectDetector("ncnn", "traffic", "CPU") != STATUS_SUCCESS)
    // if(this->m_detector->initObjectDetector("snpe", "traffic", "CPU") != STATUS_SUCCESS)
    {
        LOG_FAIL("Init detector failed");  
        return STATUS_FAILED;          
    }
    return STATUS_SUCCESS;
}
int Intrusion::set(settingsIntrusion settings)
{
    this->m_settings = settings;
    return STATUS_SUCCESS;
}
int Intrusion::update(Mat& frame, vector<outDataIntrusion>& outData)
{
    try
    {
        Mat img;
        frame.copyTo(img);
        int widthFrame = img.cols;
        int heightFrame = img.rows;
        outData.clear();

        this->m_flIsNewOutSide = false;
        this->m_flIsNewAccessing = false;      
        // draw region intrusion
        for( int i = 0; i < this->m_settings.arRegionsSet.size(); i++)
        {
            line( img, this->m_settings.arRegionsSet[i],  
            this->m_settings.arRegionsSet[(i+1) % this->m_settings.arRegionsSet.size()], Scalar( 0, 0, 200), 2, LINE_AA );
        }
        std::vector<std::vector<cv::Point> > fillContAll;
        fillContAll.push_back(this->m_settings.arRegionsSet);
        cv::Mat layer = cv::Mat::zeros(img.size(), CV_8UC3);
        cv::fillPoly(layer, fillContAll, Scalar( 0, 0, 100));
        cv::addWeighted(img, 1, layer, 0.3, 0, img);

        // detect
        std::vector<ObjectTrace> detected;
		if(this->m_detector->executeObjectDetector(img, detected, THRES_DETECT_VEHICLE) != STATUS_SUCCESS)
        {
            LOG_FAIL("Execute Anpr detector failed");
            return STATUS_FAILED;
        }
        auto start = std::chrono::high_resolution_clock::now();    
        
        // tracking
        vector<TrackingTrace> tracks;
        this->m_tracker->process(detected, tracks);
        
        //delete object which abandoned
        for(auto it = this->m_listTracked.begin(); it != this->m_listTracked.end();)
        {
            const int theId =  (*it).track_id;
            const auto p = find_if(tracks.begin(), tracks.end(), 
                                        [theId] ( const TrackingTrace& a ) { return (a.m_ID == theId);}); 
            if (p == tracks.end() && it != this->m_listTracked.end())
                it = this->m_listTracked.erase(it);                
            else 
                it++;
        }

        // check opposition
        for (auto &track: tracks) 
        {           
            if(!track.isOutOfFrame)
            {
                // cv::Mat imgVehivle_ = img(track.m_rect).clone();
                // std::string colorV = this->m_ColorClassifier.ClassifyColor(imgVehivle_);

                // if object is not in list of objects, then abort (skip)
                if(find(this->m_settings.arListObjects.begin(), this->m_settings.arListObjects.end(), track.m_type) 
                == this->m_settings.arListObjects.end())
                    continue;
                // draw box
                // rectangle(img, track.m_rect, Scalar(255, 255, 255), 1, 8);
                // char text[100];
			    // sprintf(text,"%d:%s,%s", (int)track.m_ID, track.m_type.c_str(),colorV.c_str());
			    // cv::putText(img, text, cv::Point(track.m_rect.x, track.m_rect.y), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1); 

                // find object in tracked list                 
                int theId = track.m_ID;
                const auto p = find_if(this->m_listTracked.begin(), this->m_listTracked.end(), [theId] (const outDataIntrusion& a) {return theId == a.track_id;});               
                // if found
                if(p != this->m_listTracked.end())
                {
                    int i = distance(this->m_listTracked.begin(), p);
                    this->m_listTracked[i].rect = track.m_rect;
                    // if object inside region
                    if(this->m_tracker->isInsideRegion(this->m_settings.arRegionsSet, 
                    cv::Point(track.m_rect.x + track.m_rect.width/2, 
                    track.m_rect.y + track.m_rect.height/2 )))
                    {
                        this->m_listTracked[i].rect = track.m_rect;
                        this->m_listTracked[i].cTimeCounter = getTickCount()/getTickFrequency() - this->m_listTracked[i].fTimeAccessed;
                    }
                    // check object has gone out of region 
                    else
                    {
                        this->m_listTracked.erase(p);
                        this->m_flIsNewOutSide = true;
                    }
                }
                //if not found => new object
                else
                {
                    // if object inside region
                    if(this->m_tracker->isInsideRegion(this->m_settings.arRegionsSet, 
                    cv::Point(track.m_rect.x + track.m_rect.width/2, 
                    track.m_rect.y + track.m_rect.height/2 )))
                    {
                        outDataIntrusion obj_in;
                        cv::Mat imgVehivle = img(track.m_rect).clone();
                        // obj_in.color = this->m_ColorClassifier.ClassifyColor(imgVehivle);
                        obj_in.track_id = track.m_ID;
                        obj_in.label = track.m_type;
                        obj_in.rect = track.m_rect;
                        obj_in.fTimeAccessed = getTickCount() / getTickFrequency();
                        this->m_listTracked.push_back(obj_in); // push to list in                    
                        this->m_flIsNewAccessing = true;
                    }                  
                }  
            }
            else
            {
                const int theId =  track.m_ID;
                const auto p = find_if(this->m_listTracked.begin(), this->m_listTracked.end(), 
                                        [theId] ( const outDataIntrusion& a ) { return (a.track_id == theId);});                         
                if (p != this->m_listTracked.end()) 
                {
                    int dist = distance(this->m_listTracked.begin(), p);
                    this->m_listTracked[dist].isOutOfFrame = true;
                }
            } 
        }
        // cout << "Object counter: " << this->m_listTracked.size()
        //     << ", Time counter: " << this->m_cTimeCounter << endl;
        frame = img;  
        if(this->m_listTracked.size() >= this->m_settings.cObjectCounter)
        {          
            this->m_cTimeCounter = getTickCount() / getTickFrequency() - this->m_fTimeStartCounter;
            if(this->m_cTimeCounter >= this->m_settings.cTimeOut && this->m_flIsReadyEvent)
            {
                outData = this->m_listTracked;
                this->m_flIsReadyEvent = false;
                this->m_flFirstEvent = true;
                // cout << "EVENT\n";
                return STATUS_SUCCESS;               
            }
            else if(this->m_cTimeCounter > 0 && this->m_flFirstEvent)
            {
                uint16_t cRepeatCounter = (this->m_cTimeCounter - this->m_settings.cTimeOut) / this->m_settings.cTimeRepeat;
                if(this->m_cRepeatCounter != cRepeatCounter)
                {
                    this->m_cRepeatCounter = cRepeatCounter;
                    if ((this->m_cTimeCounter - this->m_settings.cTimeOut) % this->m_settings.cTimeRepeat == 0)
                    {                                                      
                        outData = this->m_listTracked;
                        this->m_flIsReadyEvent = false;
                        // cout << "EVENT REPATED: " << this->m_cRepeatCounter << endl;
                        return STATUS_SUCCESS;             
                    } 
                }
            }                    
        }
        else
        {
            this->m_flIsReadyEvent = true;    
            this->m_cRepeatCounter = 1;      
            this->m_fTimeStartCounter = getTickCount() / getTickFrequency();
            this->m_cTimeCounter = getTickCount() / getTickFrequency() - this->m_fTimeStartCounter;           
        }
        // // having new accessing
        if(this->m_flIsNewAccessing)
        {
            if(this->m_listTracked.size() >= this->m_settings.cObjectCounter) 
            {
                if(this->m_cTimeCounter >= this->m_settings.cTimeOut) 
                {
                    outData = this->m_listTracked; // get new event
                    this->m_cTimeCounter = getTickCount() / getTickFrequency() - this->m_fTimeStartCounter;
                    return STATUS_SUCCESS;
                }    
            }
        }
        auto end = std::chrono::high_resolution_clock::now();    
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		cout << "Time track: " << duration.count() << endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return STATUS_SUCCESS;
}