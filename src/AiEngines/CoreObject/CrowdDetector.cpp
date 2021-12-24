

/*
    IntrusionDetector.hpp
    Author: ThangLMb
    Date: May 12th, 2021
*/
#include "CrowdDetector.hpp"

CrowdDetector::CrowdDetector()
{
    this->tracking = new ObjectTracking();
    this->objNames = loadObjectNames(PATH_LABEL);
    this->listTrack.clear();
}
CrowdDetector::~CrowdDetector()
{
    delete this->tracking;
}
void CrowdDetector::setCrowd(vector<Point> regionsSet, crowdSet crowd_set )
{
    this->regionsSet = regionsSet;
    this->crowd_set = crowd_set;
}
void CrowdDetector::updateCrowd(Mat& frame, vector<bbox_t> detected, vector<outDataIntrusion>& output)
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

            printf("Object Tracking: %d\n", tracks.size() );
            for (outDataIntrusion &track: this->listTrack) 
            {
                track.isOutOfFrame = true;
            }

            // draw box intrusion
            for( int i = 0; i < this->regionsSet.size(); i++)
            {
                line( in_frame, this->regionsSet[i],  this->regionsSet[(i+1) % this->regionsSet.size()], Scalar( 0, 0, 200), 2, LINE_AA );
            }
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
                        int i = distance(this->listTrack.begin(), p);
                        // check object has gone into region 
                        if(this->tracking->isInsideRegion(this->regionsSet,cv::Point(track.m_brect.boundingRect().x + track.m_brect.boundingRect().width/2, 
                        track.m_brect.boundingRect().y + track.m_brect.boundingRect().height/2 )))
                        {
                            // rectangle(in_frame, track.m_brect.boundingRect(), Scalar(0, 255, 0), 3, 8);
                            this->listTrack[i].isOutOfFrame = false;
                            this->listTrack[i].rect = track.m_brect.boundingRect();
                            this->listTrack[i].list_rect.push_back(track.m_brect.boundingRect());
                            this->listTrack[i].total_time = getTickCount()/freq - this->listTrack[i].stick_in;
                        }
                        // check object has gone out of region 
                        else
                        {
                            this->listTrack.erase(p);
                        }
                    }
                    else 
                    {
                        // if object inside region
                        if(this->tracking->isInsideRegion(this->regionsSet,cv::Point(track.m_brect.boundingRect().x + track.m_brect.boundingRect().width/2, 
                        track.m_brect.boundingRect().y + track.m_brect.boundingRect().height/2 )))
                        {
                            outDataIntrusion obj_in;
                            obj_in.track_id = track.m_ID;
                            obj_in.list_rect.push_back(track.m_brect.boundingRect());
                            obj_in.rect = track.m_brect.boundingRect();
                            obj_in.obj_status = OBJ_STATUS::INSIDE;
                            obj_in.stick_in = getTickCount()/freq;
                            obj_in.isEvent = true;
                            this->listTrack.push_back(obj_in); // push to list in
                            // rectangle(in_frame, track.m_brect.boundingRect(), Scalar(0, 255, 0), 3, 8);
                        } 
                    }          
                }
            }
        }

        // define crowd event
        int time_inside = 0;
        for (outDataIntrusion &track: this->listTrack) 
        {
            time_inside += track.total_time;
        }
        if(this->listTrack.size() > 0)
        {
            int time_count = (int)(time_inside/this->listTrack.size()) ;
            bool theId_Event = true;
            const auto obj_event = find_if(this->listTrack.begin(), this->listTrack.end(), [theId_Event] ( const outDataIntrusion& a ) { return a.isEvent == theId_Event;});
            if(this->listTrack.size() >= this->crowd_set.max_object && 
            time_count >= this->crowd_set.timeout &&
            obj_event != this->listTrack.end())
            {
                printf("Crowd Detect\n");
                for(int k = 0; k < this->listTrack.size(); k++)
                {
                    rectangle(in_frame, listTrack[k].rect, Scalar(0, 255, 0), 3, 8);
                    this->listTrack[k].isEvent = false;
                }

            output = this->listTrack;
        }

        }
       
        frame = in_frame;
    }
    catch(const exception& e) {
        cerr << e.what() << '\n';
    }  
}