/*
    Module: AnprRecognizer.cpp
    Author: ThangLmb
    Date: June 24, 2021
*/
#include "AnprRecognizer.hpp"

AnprRecognizer::AnprRecognizer(/* args */)
{
    this->detector = new AnprDetector();
    this->tracker = new ObjectTracking();
    this->listPlateTracks.clear();
}

AnprRecognizer::~AnprRecognizer()
{
    if(this->ocrVN != nullptr)  delete this->ocrVN;

    if(this->ocrUS != nullptr)  delete this->ocrUS;

    if(this->ocrMalay != nullptr)  delete this->ocrMalay;

    if(this->detector != nullptr) delete this->detector;

    if(this->tracker != nullptr) delete this->tracker;
}
int AnprRecognizer::init(Nations nation)
{
    this->nations = nation;
    // init detector number plate ( VN, US, ...)
    if(!this->detector->init(nation))
            return STATUS_FAILED;

    // init ocr
    if(nation == Nations::VN)
    {
        std::cout << "OCR model: " << DIR_OCR_DET_VN << ", " << DIR_OCR_RECOG_VN << endl;
        if(initVn(DIR_OCR_DET_VN, DIR_OCR_RECOG_VN) != STATUS_SUCCESS)
            return STATUS_FAILED;
        
    }
    else if(nation == Nations::US)   
    {
        std::cout << "OCR model: " << DIR_OCR_DET_US << ", " << DIR_OCR_RECOG_US << endl;
        if(initUS(DIR_OCR_DET_US, DIR_OCR_RECOG_US) != STATUS_SUCCESS)
            return STATUS_FAILED;
    }    
    else if(nation == Nations::MALAY)   
    {
        std::cout << "OCR model: " << DIR_OCR_DET_MALAY << ", " << DIR_OCR_RECOG_MALAY << endl;
        if(initMalay(DIR_OCR_DET_MALAY, DIR_OCR_RECOG_MALAY) != STATUS_SUCCESS)
            return STATUS_FAILED;
    } 
    return STATUS_SUCCESS;
}

int AnprRecognizer::initVn(std::string pathDet, std::string pathRecog)
{
    this->ocrVN = new OcrConfig;
    this->ocrVN->configOCR["width_det"]  = 192;
    this->ocrVN->configOCR["height_det"]  = 64;
    this->ocrVN->configOCR["width_recog"]  = 100;
    this->ocrVN->configOCR["height_recog"]  = 32;
    this->ocrVN->configOCR["det_db_thresh"]  = 0.3;
    this->ocrVN->configOCR["det_db_box_thresh"]  = 0.5;
    this->ocrVN->configOCR["det_db_unclip_ratio"]  = 1.6;
    this->ocrVN->configOCR["det_db_use_dilate"]  = 0;
    this->ocrVN->configOCR["det_use_polygon_score"]  = 1;
    this->ocrVN->configOCR["use_direction_classify"]  = 0;

    this->ocrVN->detector = loadModel(pathDet);
    this->ocrVN->recog = loadModel(pathRecog);

    this->ocrVN->dict = ReadDict(PATH_DICTIONARY);
    this->ocrVN->dict.insert(this->ocrVN->dict.begin(), "#"); // blank char for ctc
    this->ocrVN->dict.push_back(" ");
    printf("[INFO] - Init OCR Vietnam successfully\n");
    return STATUS_SUCCESS;
}

int AnprRecognizer::initUS(std::string pathDet, std::string pathRecog)
{
    this->ocrUS = new OcrConfig;

    this->ocrUS->configOCR["width_det"]  =128;
    this->ocrUS->configOCR["height_det"]  = 96;
    this->ocrUS->configOCR["width_recog"]  = 100;
    this->ocrUS->configOCR["height_recog"]  = 32;
    this->ocrUS->configOCR["det_db_thresh"]  = 0.3;
    this->ocrUS->configOCR["det_db_box_thresh"]  = 0.5;
    this->ocrUS->configOCR["det_db_unclip_ratio"]  = 1.6;
    this->ocrUS->configOCR["use_direction_classify"]  = 0;
    
    this->ocrUS->detector = loadModel(pathDet);
    this->ocrUS->recog = loadModel(pathRecog);

    this->ocrUS->dict = ReadDict(PATH_DICTIONARY);
    this->ocrUS->dict.insert(this->ocrUS->dict.begin(), "#"); // blank char for ctc
    this->ocrUS->dict.push_back(" ");
    printf("[INFO] - Init model text recognition in US successfully\n");
    return STATUS_SUCCESS;
}

int AnprRecognizer::initMalay(std::string pathDet, std::string pathRecog)
{
    this->ocrMalay = new OcrConfig;

    this->ocrMalay->configOCR["width_det"]  = 128;
    this->ocrMalay->configOCR["height_det"]  = 96;
    this->ocrMalay->configOCR["width_recog"]  = 100;
    this->ocrMalay->configOCR["height_recog"]  = 32;
    this->ocrMalay->configOCR["det_db_thresh"]  = 0.3;
    this->ocrMalay->configOCR["det_db_box_thresh"]  = 0.5;
    this->ocrMalay->configOCR["det_db_unclip_ratio"]  = 1.6;
    this->ocrMalay->configOCR["use_direction_classify"]  = 0;
    
    this->ocrMalay->detector = loadModel(pathDet);
    this->ocrMalay->recog = loadModel(pathRecog);
    this->ocrMalay->dict = ReadDict(PATH_DICTIONARY);
    this->ocrMalay->dict.insert(this->ocrMalay->dict.begin(), "#"); // blank char for ctc
    this->ocrMalay->dict.push_back(" ");
    printf("[INFO] - Init model text recognition in Malaysia successfully\n");
    return STATUS_SUCCESS;
}

std::string AnprRecognizer::readText( cv::Mat& img, Nations nation, float& confidence)
{
    std::string textOut = "";
    if(!img.empty())
    {
        std::vector<std::string> rec_text;
        std::vector<float> rec_text_score;
        if(nation == Nations::VnSquare)
        {             
            this->ocrVN->configOCR["width_det"]  = 128;
            this->ocrVN->configOCR["height_det"]  = 96;
            std::vector<std::vector<std::vector<int>>> boxes = RunDetModel(this->ocrVN->detector, img, this->ocrVN->configOCR); 
            RunRecModel(boxes, img, this->ocrVN->recog, rec_text, rec_text_score, this->ocrVN->dict, this->ocrVN->configOCR);
        }
        else if(nation == Nations::VnRect)
        {      
            this->ocrVN->configOCR["width_det"]  = 128;
            this->ocrVN->configOCR["height_det"]  = 64;
            std::vector<std::vector<std::vector<int>>> boxes = RunDetModel(this->ocrVN->detector, img, this->ocrVN->configOCR);
            RunRecModel(boxes, img, this->ocrVN->recog, rec_text, rec_text_score, this->ocrVN->dict, this->ocrVN->configOCR);
        }
        else if(nation == Nations::US)
        {
            std::vector<std::vector<std::vector<int>>> boxes = RunDetModel(this->ocrUS->detector, img, this->ocrUS->configOCR);
            RunRecModel(boxes, img, this->ocrUS->recog, rec_text, rec_text_score, this->ocrUS->dict, this->ocrUS->configOCR);
            // Visualization(img, boxes);
        }
        else if(nation == Nations::MALAY)
        {
            std::vector<std::vector<std::vector<int>>> boxes = RunDetModel(this->ocrMalay->detector, img, this->ocrMalay->configOCR);
            RunRecModel(boxes, img, this->ocrMalay->recog, rec_text, rec_text_score, this->ocrMalay->dict, this->ocrMalay->configOCR);
        }

        float score = 0;
        std::regex vnRect5("[0-9]{2}[A-Z][0-9]{5}");
        std::regex vnRect4("[0-9]{2}[A-Z][0-9]{4}");
        std::regex VnSquareBelowLine5("[0-9]{5}");
        if(rec_text.size() > 0)
        { 
            for (int i = 0; i < rec_text.size(); i++)
            {                       
                if(!isnan(rec_text_score[i]))
                    score += rec_text_score[i];
                // add '-', '.' to anpr Vietnam
                // full text license
                if(this->nations == Nations::VN)
                {
                    if(i==0)
                    {
                        if(std::regex_match(rec_text[i], vnRect5))
                        {
                            rec_text[i].insert(3, "-");
                            rec_text[i].insert(7, ".");
                        }
                        else if(std::regex_match(rec_text[i], vnRect4))
                        {
                            rec_text[i].insert(3, "-");
                        } 
                    }              
                    if(i >= 1)
                    {
                        if(std::regex_match(rec_text[i], VnSquareBelowLine5))
                        {
                            rec_text[i].insert(3, ".");
                        } 
                    }
                    textOut += "-" + rec_text[i];
                }
                else
                {
                    textOut += " " + rec_text[i];
                }           
            }
            
            textOut.erase(textOut.begin() + 0);
            confidence = score / rec_text.size();
            if(this->nations == Nations::VN)
            {   
                std::regex validLicense ("[0-9]{2}[A-Z]");
                if(!std::regex_match(textOut.begin(), textOut.begin() + 3, validLicense))
                    return "Unknown";
                if(textOut.length() < 5 || textOut.length() > 12)
                    return "Unknown";
            }
            
        }
        else 
        {
            textOut = "Unknown";
            confidence = 0;
        }
    }
    
    return textOut;
}
int CalcBlurryImage(cv::Mat matInput)
{
	cv::Mat gray ;
    cv::cvtColor(matInput, gray, cv::COLOR_BGR2GRAY);
	cv::Mat laplacianImage;
	cv::Laplacian(gray, laplacianImage, CV_64F);
	cv::Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
	cv::meanStdDev(laplacianImage, mean, stddev, cv::Mat());
	double variance = stddev.val[0] * stddev.val[0];
	return variance;
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
            std::cout << "Detected object: " << objPlates.size() << endl;
            auto end = std::chrono::high_resolution_clock::now();    
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            cout << "Time detect: " <<  duration.count() << endl;
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
                    if(objPlates[i].label == "VnRect" || objPlates[i].label == "VN_rectangle" )     
                    {
                        plate.license = readText(imgPlate, Nations::VnRect, confidence);
                    }          
                    else if(objPlates[i].label == "VnSquare" || objPlates[i].label == "VN_square")
                    {
                        plate.license = readText(imgPlate, Nations::VnSquare, confidence);
                    }           
                    if(objPlates[i].label == "US")
                    {
                        plate.license = readText(imgPlate, Nations::US, confidence) ; 
                    }
                    else if(objPlates[i].label == "Malay") // malay
                    {
                        plate.license = readText(imgPlate, Nations::MALAY, confidence) ; 
                    }
                    if(!isnan(confidence) && !isnan(objPlates[i].score))
                        plate.score = (confidence + objPlates[i].score) / 2.0;
                    else plate.score = 0;
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
        return STATUS_FAILED; 
    }
    return STATUS_SUCCESS;
}

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
        return STATUS_FAILED; 
    }
    return STATUS_SUCCESS;
}

bool AnprRecognizer::isValidPlate(cv::Mat& img)
{
    if(img.rows * img.cols >= 100) 
        return true;
    return false;
}