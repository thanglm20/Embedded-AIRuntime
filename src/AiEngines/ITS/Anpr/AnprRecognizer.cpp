/*
    Module: AnprRecognizer.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#include "AnprRecognizer.hpp"
#include "PaddleOCR-Linux-x64/PaddleOCRx64.hpp"

namespace airuntime{
    namespace aiengine{

AnprRecognizer::AnprRecognizer(Nations nation)
{
    this->m_nations = nation;
    this->detector = new AnprDetector(this->m_nations);
    this->tracker = new ObjectTracking();
    

    this->listPlateTracks.clear();
}

AnprRecognizer::~AnprRecognizer()
{
    if(this->detector != nullptr) delete this->detector;

    if(this->tracker != nullptr) delete this->tracker;
}
int AnprRecognizer::init()
{
    #ifdef ANDROID 
        ;
    #else
    this->m_licenseOcr = new PaddleOCRx64(); 
    this->m_licenseOcr->init();
    #endif
    return STATUS::SUCCESS;
}


int AnprRecognizer::recognize( cv::Mat& img, std::vector<PlateInfor>& plates)
{
    if(!img.empty())
    {
        try
        {
            std::vector<ObjectTrace> objPlates;
            auto start = std::chrono::high_resolution_clock::now();
            this->detector->detect(img, objPlates);
            
            plates.clear();
        
            for(int i = 0; i < objPlates.size(); i++)
            {
                objPlates[i].rect.x -=  objPlates[i].rect.width * 0.1;
                objPlates[i].rect.x = objPlates[i].rect.x >= 0 ? objPlates[i].rect.x : 0;

                objPlates[i].rect.y -=  objPlates[i].rect.height * 0.1;
                objPlates[i].rect.y = objPlates[i].rect.y >= 0 ? objPlates[i].rect.y : 0;

                objPlates[i].rect.width +=  objPlates[i].rect.width * 0.3;
                objPlates[i].rect.width = (objPlates[i].rect.x + objPlates[i].rect.width) > img.cols
                ?
                (img.cols - objPlates[i].rect.x) : objPlates[i].rect.width;

                objPlates[i].rect.height +=  objPlates[i].rect.height * 0.3;
                objPlates[i].rect.height = (objPlates[i].rect.y + objPlates[i].rect.height) > img.rows
                ?
                (img.rows - objPlates[i].rect.y) : objPlates[i].rect.height;

                cv::Mat imgPlate = img(objPlates[i].rect).clone();
                float confidence = 0;
                PlateInfor plate;            
                if(isValidPlate(imgPlate))
                {
                    plate.license = this->m_licenseOcr->recognize(imgPlate);
                }
                else
                {
                    plate.license = "Unknown";
                    plate.score = 0;
                }
                plate.imgPlate = imgPlate;
                plate.isNewEvent = true;
                plate.rect = objPlates[i].rect;
                plate.typePlate = objPlates[i].label;       
                plates.push_back(plate);
            }           
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    else
    {
        LOG_FAIL("Execute Anpr recognizer failed, please check your input");
        return STATUS::FAIL; 
    }
    return STATUS::SUCCESS;
}
/*
int AnprRecognizer::trackAnpr(Mat &img, std::vector<PlateInfor>& plates)
{
    plates.clear();
    if(!img.empty())
    {
        try
        {   
            // detect
            std::vector<ObjectTrace> objPlates;
            this->detector->detect(img, objPlates);
            std::cout << "Detected object: " << objPlates.size() << endl;
            // adding size of plate
            for(int i = 0; i < objPlates.size(); i++)
            {
                
                objPlates[i].rect.x -=  objPlates[i].rect.width * 0.1;
                objPlates[i].rect.x = objPlates[i].rect.x >= 0 ? objPlates[i].rect.x : 0;

                objPlates[i].rect.y -=  objPlates[i].rect.height * 0.1;
                objPlates[i].rect.y = objPlates[i].rect.y >= 0 ? objPlates[i].rect.y : 0;

                objPlates[i].rect.width +=  objPlates[i].rect.width * 0.3;
                objPlates[i].rect.width = (objPlates[i].rect.x + objPlates[i].rect.width) > img.cols
                ?
                (img.cols - objPlates[i].rect.x) : objPlates[i].rect.width;

                objPlates[i].rect.height +=  objPlates[i].rect.height * 0.3;
                objPlates[i].rect.height = (objPlates[i].rect.y + objPlates[i].rect.height) > img.rows
                ?
                (img.rows - objPlates[i].rect.y) : objPlates[i].rect.height;

            } 

            // process tracking
            std::vector<TrackingTrace> tracks;
            this->tracker->process(objPlates, tracks);

            //delete object which is abandoned
            for(auto it = this->listPlateTracks.begin(); it != this->listPlateTracks.end();)
            {
                const int theId =  (*it).track_id;
                const auto p = find_if(tracks.begin(), tracks.end(), 
                                            [theId] ( const TrackingTrace& a ) { return (a.m_ID == theId);}); 
                if (p == tracks.end() && it != this->listPlateTracks.end())
                    it = this->listPlateTracks.erase(it);                
                else 
                    it++;
            }

            // process new track and update old track
            for(auto track : tracks)
            {
                if(!track.isOutOfFrame)
                {
                    const int theId =  track.m_ID;
                    const auto p = find_if(this->listPlateTracks.begin(), this->listPlateTracks.end(), 
                                            [theId] ( const PlateInfor& a ) { return (a.track_id == theId);});  
                    
                    if (p != this->listPlateTracks.end()) // if find plate in list tracked plate
                    {
                        int dist = distance(this->listPlateTracks.begin(), p);
                        this->listPlateTracks[dist].rect = track.m_rect;
                        this->listPlateTracks[dist].isOutOfFrame = false;
                        if(this->listPlateTracks[dist].license == "Unknown" && this->listPlateTracks[dist].countUnknown <= MAX_COUNT_UNKNOWN)
                        {
                            this->listPlateTracks[dist].isNewEvent = true;
                            cv::Mat imgPlate = img(this->listPlateTracks[dist].rect).clone();
                            float confidence = 0;
                            this->listPlateTracks[dist].imgPlate = imgPlate;
                            if(isValidPlate(imgPlate))
                            {
                                this->listPlateTracks[dist].isNewEvent = true;
                                if(track.m_type == "VnRect" || track.m_type == "VN_rectangle" )     
                                {
                                    this->listPlateTracks[dist].license = readText(imgPlate, Nations::VnRect, confidence);
                                }          
                                else if(track.m_type == "VnSquare" || track.m_type == "VN_square")
                                {
                                    this->listPlateTracks[dist].license = readText(imgPlate, Nations::VnSquare, confidence);
                                }           
                                else if(track.m_type == "US")
                                {
                                    this->listPlateTracks[dist].license = readText(imgPlate, Nations::VnSquare, confidence) ; 
                                } 
                                else if(track.m_type == "Malay")
                                {
                                    this->listPlateTracks[dist].license = readText(imgPlate, Nations::MALAY, confidence) ; 
                                } 
                                if(!isnan(confidence))
                                    this->listPlateTracks[dist].score = confidence;
                                else this->listPlateTracks[dist].score = 0;

                                if(this->listPlateTracks[dist].license ==  "Unknown")
                                {
                                    this->listPlateTracks[dist].countUnknown++;
                                    this->listPlateTracks[dist].isNewEvent = false;
                                }
                            }
                            else
                            {
                                this->listPlateTracks[dist].license = "Unknown";
                                this->listPlateTracks[dist].score = 0;
                                this->listPlateTracks[dist].countUnknown++;
                                this->listPlateTracks[dist].isNewEvent = false;
                            }                            
                        }
                        else
                        {
                            this->listPlateTracks[dist].isNewEvent = false;
                        } 
                        
                    }
                    else  // if find new plate
                    {
                        cv::Mat imgPlate = img(track.m_rect).clone();
                        float confidence = 0;
                        PlateInfor plate;
                        plate.imgPlate = imgPlate;
                        plate.isNewEvent = true;
                        plate.track_id = track.m_ID;
                        plate.rect = track.m_rect;
                        plate.typePlate = track.m_type;   
                        plate.isOutOfFrame = false;
                        if(isValidPlate(imgPlate))
                        {
                            if(track.m_type == "VnRect" || track.m_type == "VN_rectangle" )     
                            {
                                plate.license = readText(imgPlate, Nations::VnRect, confidence);
                            }          
                            else if(track.m_type == "VnSquare" || track.m_type == "VN_square")
                            {
                                plate.license = readText(imgPlate, Nations::VnSquare, confidence);
                            }           
                            else if(track.m_type == "US")
                            {
                                plate.license = readText(imgPlate, Nations::VnSquare, confidence) ; 
                            } 
                            else if(track.m_type == "Malay")
                            {
                                plate.license = readText(imgPlate, Nations::MALAY, confidence) ; 
                            } 
                            if(!isnan(confidence))
                                plate.score = confidence;
                            else plate.score = 0;
                        }
                        else
                        {
                            plate.license = "Unknown";
                            plate.score = 0;
                            plate.countUnknown = 1;
                        } 
                        this->listPlateTracks.push_back(plate);
                    }
                }
                else
                {
                    const int theId =  track.m_ID;
                    const auto p = find_if(this->listPlateTracks.begin(), this->listPlateTracks.end(), 
                                            [theId] ( const PlateInfor& a ) { return (a.track_id == theId);});                         
                    if (p != this->listPlateTracks.end()) 
                    {
                        int dist = distance(this->listPlateTracks.begin(), p);
                        this->listPlateTracks[dist].isOutOfFrame = true;
                        
                    }
                } 
            }  
            //get list of output plates
            for(auto p : this->listPlateTracks)
            {
                if(p.isOutOfFrame != true) plates.push_back(p);
            }                  
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }
    else
    {
        LOG_FAIL("Execute Anpr recognizer failed, please check your input");
        return STATUS::INVALID_ARGS;
    }
    return STATUS::SUCCESS;
}
*/
bool AnprRecognizer::isValidPlate(cv::Mat& img)
{
    if(img.rows * img.cols >= 100) 
        return true;
    return false;
}

    } // aiengine
} // airuntime