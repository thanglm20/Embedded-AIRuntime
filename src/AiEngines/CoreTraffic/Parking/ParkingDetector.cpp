#include "ParkingDetector.hpp"

ParkingDetector::ParkingDetector() {
    this->tracking = new ObjectTracking();
    this->objNames = loadObjectNames("../model/traffic/traffic.names");
    this->listTracker.clear();
    this->vehicleList.clear();
}

ParkingDetector::~ParkingDetector() {

}

void ParkingDetector::setData(vector<vector<Point>> regions, vector<string> label, int input) {
    if ((regions.size() != 0) && (label.size() == regions.size())) {
        this->regionsSet = regions;
        this->labelSet = label;
    }
    this->timeP = input;
}

void ParkingDetector::update(Mat& frame, vector<bbox_t> data, vector<OutputData>& output) {
    try {
        Mat Frame;
        frame.copyTo(Frame);

        if (this->regionsSet.size() == 0) return;
        vector<bbox_t> detected;
        for (int index = 0; index < this->regionsSet.size(); index++) {
            vector<string> labels;
            labels.clear();
            istringstream f(this->labelSet[index]);
            string s;
            while (getline(f, s, ',')) {
                labels.push_back(this->objNames[stoi(s)]);
            }
            for (auto &i: data) {
                Point2f cBox = Point2f(i.x + float(i.w/2), i.y + float(i.h/2));
                if (find(labels.begin(), labels.end(), this->objNames[i.obj_id]) != labels.end()) {
                    if (pointPolygonTest(this->regionsSet[index], cBox, false) >= 0) {
                        detected.push_back(i);
                    }
                }
            }
        }
        regions_t regions;
        bbox2regions(detected, regions, this->objNames);
        vector<TrackingObject> tracks;
        tracking->process(regions, frame, tracks);
        double freq = getTickFrequency();
        int64 current = getTickCount()/freq;
        vector<bool> checkBox;
        checkBox.clear();
        for (size_t i = 0; i < this->vehicleList.size(); ++i) checkBox.push_back(false);
        for (auto& track: tracks) {
            if (track.IsRobust(cvRound(2), 0.8f, Size2f(0.1f, 8.0f))) {  
                Rect smallbox;
                smallbox.x = track.m_brect.boundingRect().x + track.m_brect.boundingRect().width/4;
                smallbox.y = track.m_brect.boundingRect().y + track.m_brect.boundingRect().height/4;
                smallbox.width = track.m_brect.boundingRect().width - track.m_brect.boundingRect().width/2;
                smallbox.height = track.m_brect.boundingRect().height - track.m_brect.boundingRect().height/2;    
                const int theId =  track.m_ID;
                const auto p = find_if(this->listTracker.begin(), this->listTracker.end(), [theId] ( const object& a ) { return a.id == theId;});
                if (checkStill(track)) {
                    if (p != this->listTracker.end()) {
                        int index = distance(this->listTracker.begin(), p);
                        bool exist = false;
                        Point2f cBox = Point2f(track.m_brect.boundingRect().x + float(track.m_brect.boundingRect().width/2), track.m_brect.boundingRect().y + float(track.m_brect.boundingRect().height/2));
                        int duration = current - this->listTracker[index].begin;
                        for (int i = 0; i < this->vehicleList.size(); i++) {
                            if (checkBox[i]) continue;
                            if (this->vehicleList[i].boxes.contains(cBox)) {
                                exist = true;
                                checkBox[i] = true;
                                this->vehicleList[i].boxes = smallbox;
                                this->vehicleList[i].time = getTickCount()/freq;
                                this->vehicleList[i].disappear = false;
                                this->vehicleList[i].kill = false;
                                this->vehicleList[i].isNew = false;
                                this->vehicleList[i].obj_id = convertType2Int(this->objNames, track.m_type);
                                break;
                            }
                        }
                        if (!exist) {
                            if (duration >= this->timeP) {
                                vehicleInfo boxinfo;
                                boxinfo.boxes = smallbox;
                                boxinfo.time = getTickCount()/freq;
                                boxinfo.disappear = false;
                                boxinfo.kill = false;
                                boxinfo.isNew = true;
                                boxinfo.id = track.m_ID;
                                boxinfo.obj_id = convertType2Int(this->objNames, track.m_type);
                                this->vehicleList.push_back(boxinfo);
                                checkBox.push_back(true);
                            }
                        }
                    }
                    else {
                        object still;
                        still.id = track.m_ID;
                        still.begin = getTickCount()/freq;
                        this->listTracker.push_back(still);
                    }
                } 
                else {
                    if (p != this->listTracker.end()) this->listTracker.erase(p);
                }
            } 
        }
        int waitTime = this->timeP;
        if (checkBox.size() != 0) {
            int i = 0;
            while (i < checkBox.size()) {
                Rect oribox;
                oribox.x = this->vehicleList[i].boxes.x - this->vehicleList[i].boxes.width/2;
                oribox.y = this->vehicleList[i].boxes.y - this->vehicleList[i].boxes.height/2;
                oribox.width = this->vehicleList[i].boxes.width*2;
                oribox.height = this->vehicleList[i].boxes.height*2;
                OutputData info;
                info.box = oribox;
                info.isDisappear = this->vehicleList[i].disappear;
                info.id = this->vehicleList[i].id;
                info.isNew = this->vehicleList[i].isNew;
                info.isKill = this->vehicleList[i].kill;
                info.obj_id = this->vehicleList[i].obj_id;
                if (!checkBox[i]) {
                    info.isNew = false;
                    this->vehicleList[i].disappear = true;
                    info.isDisappear = this->vehicleList[i].disappear;
                    if (current - this->vehicleList[i].time > waitTime) {
                        this->vehicleList[i].kill = true;
                        info.isKill = this->vehicleList[i].kill;
                        this->vehicleList.erase(this->vehicleList.begin() + i);
                        checkBox.erase(checkBox.begin() + i);
                    }
                    else {
                        i++;
                    }
                }
                else {
                    i++;
                }
                output.push_back(info);
                this->vehicleList[i].isNew = false;
            }
        }
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
}

int ParkingDetector::convertType2Int(vector<string>& objNames, string type) {
    for (int index = 0; index <objNames.size(); index++) {
        if (type == objNames[index]) return index;
    }
    return 0;
}

bool ParkingDetector::checkStill(TrackingObject tracker) {
    constexpr size_t minSize = 10;
    if (tracker.m_trace.size() > minSize) {
        Rect smallbox;
        smallbox.x = tracker.m_brect.boundingRect().x + tracker.m_brect.boundingRect().width*3/8;
        smallbox.y = tracker.m_brect.boundingRect().y + tracker.m_brect.boundingRect().height*3/8;
        smallbox.width = tracker.m_brect.boundingRect().width/4;
        smallbox.height = tracker.m_brect.boundingRect().height/4;

        const TrajectoryPoint &pt = tracker.m_trace.at(tracker.m_trace.size()-minSize);
        return smallbox.contains(Point(pt.m_prediction.x, pt.m_prediction.y));
    }
    return false;
}