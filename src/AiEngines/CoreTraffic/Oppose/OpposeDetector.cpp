#include <fstream>
#include "OpposeDetector.hpp"

#define THRESHDIST 0.033

OpposeDetector::OpposeDetector() {
    this->tracking = new ObjectTracking();
    this->objNames = loadObjectNames("../model/traffic/traffic.names");
    this->listTrack.clear();
    
}

OpposeDetector::~OpposeDetector() {
    delete this->tracking;
}

void OpposeDetector::setData(vector<vector<Point>> regions, vector<string> label) {
    
    if ((regions.size() != 0) && (label.size() == regions.size())) {
        this->regionsSet = regions;
        this->labelSet = label;
    }
}
static int iCountForward = 0;
static int iCountReverse = 0;
void OpposeDetector::update(Mat &frame, vector<bbox_t> detected, vector<OutputData>& output) {
    try {
        Mat Frame;
        frame.copyTo(Frame);

        output.clear();
        int widthFrame = frame.cols;
        int heightFrame = frame.rows;
        if (this->regionsSet.size() == 0) return;
        if (detected.size() != 0) {
            std::vector<bbox_t> subDetect;
            for (auto &i: detected) subDetect.push_back(i);
            regions_t regions;
            bbox2regions(subDetect, regions, this->objNames);
            vector<TrackingObject> tracks;
            this->tracking->process(regions, frame, tracks);
            
            double freq = getTickFrequency();
            for (Track &track: this->listTrack) track.outOfTheFrame = true;
            for (int index = 0; index < this->regionsSet.size(); index++) {
                vector<string> labels;
                labels.clear();
                istringstream f(this->labelSet[index]);
                string s;
                while (getline(f, s, ',')) {
                    labels.push_back(this->objNames[stoi(s)]);
                }
                
                Point2f begin1((float) this->regionsSet[index][0].x/widthFrame, (float) this->regionsSet[index][0].y/heightFrame);
                Point2f begin2((float) this->regionsSet[index][1].x/widthFrame, (float) this->regionsSet[index][1].y/heightFrame);
                line(Frame, Point(this->regionsSet[index][0].x, this->regionsSet[index][0].y), 
                    Point(this->regionsSet[index][1].x, this->regionsSet[index][1].y), Scalar( 0, 255, 0), 1, LINE_AA);
                Point2f end1((float) this->regionsSet[index][2].x/widthFrame, (float) this->regionsSet[index][2].y/heightFrame);
                Point2f end2((float) this->regionsSet[index][3].x/widthFrame, (float) this->regionsSet[index][3].y/heightFrame);
                line(Frame, Point(this->regionsSet[index][2].x, this->regionsSet[index][2].y), 
                    Point(this->regionsSet[index][3].x, this->regionsSet[index][3].y), Scalar( 0, 0, 255), 1, LINE_AA);

                // Point2f end1((float) this->regionsSet[index][0].x/widthFrame, (float) this->regionsSet[index][0].y/heightFrame);
                // Point2f end2((float) this->regionsSet[index][1].x/widthFrame, (float) this->regionsSet[index][1].y/heightFrame);
                // line(Frame, Point(this->regionsSet[index][0].x, this->regionsSet[index][0].y), 
                //     Point(this->regionsSet[index][1].x, this->regionsSet[index][1].y), Scalar( 0, 0, 255), 1, LINE_AA);
                // Point2f begin1((float) this->regionsSet[index][2].x/widthFrame, (float) this->regionsSet[index][2].y/heightFrame);
                // Point2f begin2((float) this->regionsSet[index][3].x/widthFrame, (float) this->regionsSet[index][3].y/heightFrame);
                // line(Frame, Point(this->regionsSet[index][2].x, this->regionsSet[index][2].y), 
                //     Point(this->regionsSet[index][3].x, this->regionsSet[index][3].y), Scalar( 0, 255, 0), 1, LINE_AA);

                cv::putText(Frame, "O",  Point(this->regionsSet[index][0].x, this->regionsSet[index][0].y) , FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
                cv::putText(Frame, "1",  Point(this->regionsSet[index][1].x, this->regionsSet[index][1].y) , FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
                cv::putText(Frame, "2",  Point(this->regionsSet[index][2].x, this->regionsSet[index][2].y) , FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
                cv::putText(Frame, "3",  Point(this->regionsSet[index][3].x, this->regionsSet[index][3].y) , FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);

                for (auto &track: tracks) {
                    //std::cout << "Size list track: " << this->listTrack.size() << endl;
                    // modified by thanglmb
                    char text[10];
                    sprintf(text,"%d", (int)track.m_ID);
                    cv::putText(Frame, text, cv::Point(track.m_brect.boundingRect().x, track.m_brect.boundingRect().y), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
                    if (track.IsRobust(cvRound(2), 0.85f, Size2f(0.1f, 8.0f)) && find(labels.begin(), labels.end(), track.m_type) != labels.end()) 
                    {
                        // std::cout <<  "Type: " << track.m_type << ", ID: " << track.m_ID << endl;
                       
                        const int theId =  track.m_ID;
                        const auto p = find_if(this->listTrack.begin(), this->listTrack.end(), [theId] ( const Track& a ) { return a.id == theId;});                       
                        if (p != this->listTrack.end()) {
                            int i = distance(this->listTrack.begin(), p);
                            this->listTrack[i].outOfTheFrame = false;
                            if (this->listTrack[i].brect.size() < 200) this->listTrack[i].brect.push_back(track.m_brect.boundingRect());
                            //if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(end1, end2, 0))) {
                            if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(begin1, begin2, 0))) {
                                rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 0, 255), 2, LINE_AA);
                                OutputData data;
                                data.id = this->listTrack[i].id;
                                data.brect = this->listTrack[i].brect;
                                data.obj_id = convertType2Int(this->objNames, track.m_type);
                                output.push_back(data);
                                this->listTrack.erase(p);
                            } 
                            rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 0, 255), 1, LINE_AA);
                        }
                        // starting on oppose
                        else {
                        //if (p == this->listTrack.end()){

                            //std::cout << "Size list track: " << this->listTrack.size() << endl;
                            //if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(begin1, begin2, 1))) {
                            int direction = 0;
                            if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(end1, end2, 1), direction)) {
                                rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 255, 0), 2, LINE_AA);
                                Track object;
                                object.id = track.m_ID;
                                object.brect.clear();
                                object.brect.push_back(track.m_brect.boundingRect());
                                object.label = track.m_type;
                                object.tick = getTickCount()/freq;
                                object.outOfTheFrame = false;
                                this->listTrack.push_back(object);
                                if(direction == 2)
                                {
                                    iCountForward++;
                                    char text1[20];
                                    sprintf(text1,": Forward: %d", iCountForward);
                                    cv::putText(Frame, text, cv::Point(10, 10), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
                                    std::cout << "Forward: " << iCountForward << endl;
                                }
                                else if(direction == 1)
                                {
                                    iCountReverse++;
                                    char text2[20];
                                    sprintf(text2,"Reverse: %d", iCountReverse);
                                    cv::putText(Frame, text, cv::Point(10, 20), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
                                    std::cout << "Reverse: " << iCountReverse << endl;
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < this->listTrack.size(); i++) if (this->listTrack[i].outOfTheFrame) this->listTrack.erase(this->listTrack.begin() + i);
        }
        
        //putText(Frame, std::to_string(this->listTrack.size()), Point(50, 50), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 1);
        char text1[20];
        sprintf(text1,"Forward: %d", iCountForward);
        cv::putText(Frame, text1, cv::Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
        char text2[20];
        sprintf(text2,"Reverse: %d", iCountReverse);
        cv::putText(Frame, text2, cv::Point(50, 100), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
    // Region mask
        // cv::Mat regionMask = cv::Mat(Frame.rows, Frame.cols, CV_8UC3, cv::Scalar(0,0,0));
        // for(auto region : this->regionsSet )
        // {
        //     fillConvexPoly(regionMask, region, cv::Scalar(255, 255, 255));
        // }
        // cv::Mat maskedFrame;
        // Frame.copyTo(maskedFrame, regionMask);
        // imshow("Oppose", maskedFrame);

        // draw lines
        // for (int index = 0; index < this->regionsSet.size(); index++)
        // {
        //     Point2f begin1((float) this->regionsSet[index][0].x/widthFrame, (float) this->regionsSet[index][0].y/heightFrame);
        //     Point2f begin2((float) this->regionsSet[index][1].x/widthFrame, (float) this->regionsSet[index][1].y/heightFrame);
        //     line(Frame, Point(this->regionsSet[index][0].x, this->regionsSet[index][0].y), 
        //         Point(this->regionsSet[index][1].x, this->regionsSet[index][1].y), Scalar( 0, 255, 0), 2, LINE_AA);
        //     Point2f end1((float) this->regionsSet[index][2].x/widthFrame, (float) this->regionsSet[index][2].y/heightFrame);
        //     Point2f end2((float) this->regionsSet[index][3].x/widthFrame, (float) this->regionsSet[index][3].y/heightFrame);
        //     line(Frame, Point(this->regionsSet[index][2].x, this->regionsSet[index][2].y), 
        //         Point(this->regionsSet[index][3].x, this->regionsSet[index][3].y), Scalar( 0, 0, 255), 2, LINE_AA);

        // }
        imshow("Oppose", Frame);
        
        //waitKey(1);
    }
    catch(const exception& e) {
        cerr << e.what() << '\n';
    }
}
