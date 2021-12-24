
/*
    IntrusionDetector.cpp
    Author: ThangLMb
    Date: May 12th, 2021
*/
#include "IntrusionDetector.hpp"

IntrusionDetector::IntrusionDetector()
{
    this->tracking = new ObjectTracking();
    this->objNames = loadObjectNames(PATH_LABEL);
    this->listTrack.clear();
}

IntrusionDetector::~IntrusionDetector()
{
    delete this->tracking;
}

void IntrusionDetector::setIntrusion(vector<Point> regionsSet, intrusionSet intrusion_set)
{
    this->regionsSet = regionsSet;
    this->intrusion_set = intrusion_set;
}

void IntrusionDetector::updateIntrusion(Mat& frame, vector<bbox_t> detected, vector<outDataIntrusion>& output)
{
    try {
        Mat in_frame;
        frame.copyTo(in_frame);

        output.clear();
        int widthFrame = in_frame.cols;
        int heightFrame = in_frame.rows;

        if (detected.size() != 0) 
        {
            std::vector<bbox_t> subDetect;
            for (auto &i: detected) subDetect.push_back(i);
            regions_t regions;
            bbox2regions(subDetect, regions, this->objNames);
            vector<TrackingObject> tracks;
            this->tracking->process(regions, in_frame, tracks);

            for (outDataIntrusion &track: this->listTrack) 
            {
                track.isOutOfFrame = true;
                track.isEvent = false;
            }

            // draw box intrusion
            for( int i = 0; i < this->regionsSet.size(); i++)           
                line( in_frame, this->regionsSet[i],  this->regionsSet[(i+1) % this->regionsSet.size()], Scalar( 0, 0, 200), 2, LINE_AA );
            std::vector<std::vector<cv::Point> > fillContAll;
            fillContAll.push_back(this->regionsSet);
            cv::Mat layer = cv::Mat::zeros(in_frame.size(), CV_8UC3);
            cv::fillPoly(layer, fillContAll, Scalar( 0, 0, 100));
            cv::addWeighted(in_frame, 1, layer, 0.3, 0, in_frame);

            double freq = getTickFrequency();
            for (auto &track: tracks) 
            {
                if (track.IsRobust(cvRound(2), 0.85f, Size2f(0.1f, 8.0f))) 
                {
                    const int theId =  track.m_ID;
                    const auto p = find_if(this->listTrack.begin(), this->listTrack.end(), [theId] ( const outDataIntrusion& a ) { return a.track_id == theId;});                       
                    if (p != this->listTrack.end())
                    {
                        if(!this->tracking->isInsideRegion(this->regionsSet,cv::Point(track.m_brect.boundingRect().x + track.m_brect.boundingRect().width/2, 
                        track.m_brect.boundingRect().y + track.m_brect.boundingRect().height/2 )))
                        {
                            // if config direction is detecting inside
                            if(this->intrusion_set.direction == "out"  || this->intrusion_set.direction == "all")
                            {
                                outDataIntrusion obj_out;
                                obj_out.track_id = track.m_ID;
                                obj_out.list_rect.push_back(track.m_brect.boundingRect());
                                obj_out.obj_status = OBJ_STATUS::OUTSIDE;
                                obj_out.stick_in = getTickCount()/freq;
                                obj_out.isEvent = true;
                                output.push_back(obj_out); // push object to event
                                rectangle(in_frame, track.m_brect.boundingRect(), Scalar(0, 255, 0), 3, 8);
                                printf("Detected object go outside\n");                             
                            }
                            this->listTrack.erase(p);
                        }   
                    }
                    else 
                    {
                        // if object insde region
                        if(this->tracking->isInsideRegion(this->regionsSet,cv::Point(track.m_brect.boundingRect().x + track.m_brect.boundingRect().width/2, 
                        track.m_brect.boundingRect().y + track.m_brect.boundingRect().height/2 )))
                        {
                            
                            outDataIntrusion obj_in;
                            obj_in.track_id = track.m_ID;
                            obj_in.list_rect.push_back(track.m_brect.boundingRect());
                            obj_in.obj_status = OBJ_STATUS::INSIDE;
                            obj_in.stick_in = getTickCount()/freq;
                            // if config direction is detecting inside
                            if(this->intrusion_set.direction == "in"  || this->intrusion_set.direction == "all")
                            {
                                obj_in.isEvent = true;
                                output.push_back(obj_in); // push to outdata event
                                rectangle(in_frame, track.m_brect.boundingRect(), Scalar(0, 255, 0), 3, 8);
                            }
                            this->listTrack.push_back(obj_in); // push to list in
                        } 
                    }            
                }
            }
        }
        frame = in_frame;
    }
    catch(const exception& e) {
        cerr << e.what() << '\n';
    }  
}
