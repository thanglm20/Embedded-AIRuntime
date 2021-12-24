/*
    Module: Object Tracking
    Author: Le Manh Thang
    Created: August 26th, 2021
*/

#include "CrosslineDetector.hpp"
#include <chrono>
CrosslineDetector::CrosslineDetector()
{   
    this->listTrack.clear();
    this->tracking = new ObjectTracking();
}
CrosslineDetector::CrosslineDetector(cv::Point beginPoint, cv::Point endPoint, std::string direction)
{
    this->beginPoint = beginPoint;
    this->endPoint = endPoint;
    this->direction = direction;
    this->listTrack.clear();
    this->tracking = new ObjectTracking();
}
CrosslineDetector::~CrosslineDetector()
{
    delete this->tracking;
}
void CrosslineDetector::setCrossline(cv::Point beginPoint, cv::Point endPoint, std::string direction) {
    
    this->direction = direction;
    this->beginPoint = beginPoint;
    this->endPoint = endPoint;
}

void CrosslineDetector::updateCrossline( Mat &frame, vector<ObjectTrace> detected, vector<outDataCrossline>& output) {
    try 
    {
        Mat in_frame;
        frame.copyTo(in_frame);
        output.clear();
        int widthFrame = in_frame.cols;
        int heightFrame = in_frame.rows;
        if (detected.size() != 0) 
        {           
                
            auto start = getTickCount();
            vector<TrackingTrace> tracks;
            this->tracking->process(detected, tracks);
            auto end = getTickCount();
            double time = (double)(end- start) / getTickFrequency();
            // draw line 
            Point2f beginCheckpoint((float)this->beginPoint.x/widthFrame, (float) this->beginPoint.y/heightFrame);
            Point2f endCheckpoint((float) this->endPoint.x/widthFrame, (float) this->endPoint.y/heightFrame);           
            line(in_frame, this->beginPoint, this->endPoint , Scalar( 0, 0, 255), 2, LINE_AA);
            for (auto &track: tracks) 
            {
                if(!track.isOutOfFrame)
                {
                    //if tracked object name was found in list of name objects
                    rectangle(in_frame, track.m_rect, Scalar(0, 255, 0), 1, 8);
                    char text[100];
                    sprintf(text,"%d:%s", (int)track.m_ID, track.m_type.c_str());
                    cv::putText(in_frame, text, cv::Point(track.m_rect.x, track.m_rect.y), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 1);
                    int direct_status = -1;
                    direct_status = this->tracking->checkCrossline(track, widthFrame, heightFrame, RoadLine(beginCheckpoint, endCheckpoint, 0)); 
                    // if(direct_status > 0)
                    // std::cout << "Status: " << track.m_ID << " Dir: " << direct_status << std::endl;
                    if(((direct_status == 2) && (this->direction == "in")) || 
                    ((direct_status == 1) && (this->direction == "out")) ||
                    ((direct_status > 0) && (this->direction == "all")))
                    {
                        line(in_frame, this->beginPoint, this->endPoint , Scalar( 0, 255, 255), 3, LINE_AA);
                        rectangle(in_frame, track.m_rect, Scalar(0, 255, 255), 5, 8);
                        outDataCrossline object;
                        object.track_id = track.m_ID;
                        object.rect = track.m_rect;
                        object.label = track.m_type;
                        object.isEvent = true;
                        output.push_back(object);
                    }    
                }          
            }
        }                     
        frame  = in_frame;
    }
    catch(const exception& e) {
        cerr << e.what() << '\n';
    }
}
