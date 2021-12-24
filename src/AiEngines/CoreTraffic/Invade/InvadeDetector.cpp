#include "InvadeDetector.hpp"

InvadeDetector::InvadeDetector() {
    this->tracking = new ObjectTracking();
    this->objNames = loadObjectNames("../model/traffic/traffic.names");
}

InvadeDetector::~InvadeDetector() {

}

void InvadeDetector::setData(vector<vector<Point>> regions, vector<string> label) {
    if ((regions.size() != 0) && (label.size() == regions.size())) {
        this->regionsSet = regions;
        this->labelSet = label;
    }
}

void InvadeDetector::update(Mat& frame, vector<bbox_t> detected, vector<OutputData>& boxes) {
    try {
        if (this->regionsSet.size() == 0) return;
        vector<bbox_t> subDetect;
        if (detected.size() != 0) {
            for (auto &i: detected) {
                if (combineObject(i.obj_id) == XE2BANH || combineObject(i.obj_id) == XE4BANH) { 
                    subDetect.push_back(i);
                }
            }

            regions_t regions;
            bbox2regions(subDetect, regions, objNames);

            vector<TrackingObject> tracks;
            this->tracking->process(regions, frame, tracks);

            for (int index = 0; index < this->regionsSet.size(); index++) {
                int lineType = stoi(this->labelSet[index]);
                struct Pointt p1 = {this->regionsSet[index][0].x, this->regionsSet[index][0].y}, q1 = {this->regionsSet[index][1].x, this->regionsSet[index][1].y};
                for (auto &track: tracks) {
                    if (track.IsRobust(cvRound(2), 0.8f, Size2f(0.1f, 8.0f))) {
                        OutputData info;
                        info.box = track.m_brect.boundingRect();
                        info.label = track.m_type;
                        info.id = static_cast<int>(track.m_ID);
                        info.score = track.m_confidence;
                        info.obj_id = convertType2Int(this->objNames, track.m_type);

                        int x = track.m_brect.boundingRect().x;
                        int y = track.m_brect.boundingRect().y;
                        int width = track.m_brect.boundingRect().width;
                        int height = track.m_brect.boundingRect().height;

                        struct Pointt pt = {x, y}, qt = {x + width, y};
                        struct Pointt pb = {x, y + height}, qb = {x + width, y + height};
                        struct Pointt pl = {x, y}, ql = {x, y + height};
                        struct Pointt pr = {x + width, y }, qr = {x + width, y + height};
                        if ( (doIntersect(p1, q1, pt, qt) && doIntersect(p1, q1, pb, qb)) || (doIntersect(p1, q1, pl, ql) && doIntersect(p1, q1, pr, qr)) ) {
                            int UP = 0, DOWN = 0, LEFT = 0, RIGHT = 0;
                            checkTraces(track.m_trace, UP, DOWN, LEFT, RIGHT);
                            switch (lineType)
                            {
                            case 0:
                                boxes.push_back(info);
                                break;
                            case 3:
                                if (DOWN > 0) {
                                    boxes.push_back(info);
                                }
                                break;
                            case 4:
                                if (UP > 0) {
                                    boxes.push_back(info);
                                }
                                break;
                            case 2:
                                if (LEFT > 0) {
                                    boxes.push_back(info);
                                }
                                break;
                            case 1:
                                if (RIGHT > 0) {
                                    boxes.push_back(info);
                                }
                                break;
                            default:
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    catch(const exception& e) {
        cerr << e.what() << '\n';
    }
}

bool InvadeDetector::onSegment(Pointt p, Pointt q, Pointt r) { 
	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && 
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y)) 
	return true; 

	return false; 
}

int InvadeDetector::orientation(Pointt p, Pointt q, Pointt r) { 
	int val = (q.y - p.y) * (r.x - q.x) - 
			(q.x - p.x) * (r.y - q.y); 
	if (val == 0) return 0; // colinear 
	return (val > 0)? 1: 2; // clock or counterclock wise 
} 

bool InvadeDetector::doIntersect(Pointt p1, Pointt q1, Pointt p2, Pointt q2) { 
	int o1 = orientation(p1, q1, p2); 
	int o2 = orientation(p1, q1, q2); 
	int o3 = orientation(p2, q2, p1); 
	int o4 = orientation(p2, q2, q1); 

	if (o1 != o2 && o3 != o4) 
		return true; 
	if (o1 == 0 && onSegment(p1, p2, q1)) return true; 
	if (o2 == 0 && onSegment(p1, q2, q1)) return true; 
	if (o3 == 0 && onSegment(p2, p1, q2)) return true; 
	if (o4 == 0 && onSegment(p2, q1, q2)) return true; 
	return false; 
} 

void InvadeDetector::checkTraces(Trace traces, int& UP , int& DOWN , int& LEFT , int& RIGHT ) {
    constexpr size_t minSize = 3;
    if (traces.size() > minSize) {
        const TrajectoryPoint &pt1 = traces.at(traces.size()-1);
        const TrajectoryPoint &pt2 = traces.at(traces.size() - minSize);

        if ((pt1.m_prediction.y > pt2.m_prediction.y)) {
            DOWN = 1;
        }
        if ((pt2.m_prediction.y > pt1.m_prediction.y)) {
            UP = 1;
        }
        if ((pt1.m_prediction.x > pt2.m_prediction.x)) {
            RIGHT = 1;
        }
        if ((pt2.m_prediction.x > pt1.m_prediction.x)) {
            LEFT = 1;
        }
    }
}
