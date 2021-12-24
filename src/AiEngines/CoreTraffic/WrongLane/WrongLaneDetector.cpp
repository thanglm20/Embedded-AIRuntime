#include "WrongLaneDetector.hpp"
#include <chrono>
#include <ctime>

WrongLaneDetector::WrongLaneDetector() {
    this->tracking = new ObjectTracking();
    this->objNames = loadObjectNames("../model/traffic/traffic.names");
}

void WrongLaneDetector::reset() {

}

void WrongLaneDetector::setData(vector<vector<Point>> regions, vector<string> label) {
    if ((regions.size() != 0) && (label.size() == regions.size())) {
        this->regionsSet = regions;
        this->labelSet = label;
    }
}

void WrongLaneDetector::update(Mat& frame, vector<bbox_t> detected, vector<OutputData>& output) {
    try {
        if (this->regionsSet.size() == 0) return;
        if (detected.size() == 0) return;
        vector<bbox_t> subDetect;
        for (auto &i: detected) subDetect.push_back(i);
        regions_t regions;
        bbox2regions(subDetect, regions, this->objNames);
        vector<TrackingObject> tracks;
        this->tracking->process(regions, frame, tracks);
        output.clear();
        for (int index = 0; index < this->regionsSet.size(); index++) {
            vector<string> labels;
            labels.clear();
            istringstream f(this->labelSet[index]);
            string s;
            while (getline(f, s, ',')) {
                labels.push_back(this->objNames[stoi(s)]);
            }
            for (TrackingObject &track: tracks) {
                if (track.IsRobust(cvRound(3), 0.8f, Size2f(0.1f, 8.0f))) {
                    if (find(labels.begin(), labels.end(), track.m_type) != labels.end()) {
                        if (pointPolygonTest(this->regionsSet[index], track.m_brect.boundingRect().tl(), false) >= 0 && pointPolygonTest(this->regionsSet[index], track.m_brect.boundingRect().br(), false) >= 0) {
                            OutputData info;
                            info.id = track.m_ID;
                            info.box = track.m_brect.boundingRect();
                            info.label = track.m_type;
                            info.obj_id = convertType2Int(this->objNames, track.m_type);
                            output.push_back(info);
                        }
                    }
                }
            }
        }
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
}