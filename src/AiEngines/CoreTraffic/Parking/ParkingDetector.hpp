#ifndef ParkingDetector_hpp
#define ParkingDetector_hpp

#include <iostream>
#include <ctime>
#include <fstream>
#include "TrafficDetector.hpp"
#include "Utility.hpp"

struct vehicleInfo {
    Rect boxes;
    int64 time;
    int id;
    bool isNew;
    bool disappear;
    bool kill;
    int obj_id;
};

struct object {
    int id;
    int64 begin;
    int64 current;
};

class ParkingDetector {
public:
    ParkingDetector();
	virtual ~ParkingDetector();
    void update(Mat& frame, vector<bbox_t> detected, vector<OutputData>& output);
    void setData(vector<vector<Point>> regionsSet, vector<string> labelSet, int input);

private:
    ObjectTracking* tracking;
    vector<string> objNames;
    vector<object> listTracker;
    int timeP;
    vector<vector<Point>> regionsSet;
    vector<string> labelSet;
    vector<vehicleInfo> vehicleList;

    bool checkStill(TrackingObject tracker);
    int convertType2Int(vector<string>& objNames, string type);
};

#endif /* ParkingDetector_hpp */
