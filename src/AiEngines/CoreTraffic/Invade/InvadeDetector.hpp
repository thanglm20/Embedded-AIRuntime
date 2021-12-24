#ifndef InvadeDetector_hpp
#define InvadeDetector_hpp

#include <opencv2/opencv.hpp>
#include "TrafficDetector.hpp"
#include "Utility.hpp"
#include <fstream>

struct Pointt { 
    int x; 
    int y; 
}; 

#define MAXDISTANCE 10

class InvadeDetector {
public:
    InvadeDetector();
    ~InvadeDetector();

    void update(Mat& frame, vector<bbox_t> detected, vector<OutputData>& boxes);
    void setData(vector<vector<Point>> regionsSet, vector<string> labels);
private:
    enum vehicleType {
        OTHER = 0,
        XE2BANH = 3,
        XE4BANH = 13
    };

    ObjectTracking* tracking;
    vector<string> objNames;
    vector<vector<Point>> regionsSet;
    vector<string> labelSet;

    bool onSegment(Pointt p, Pointt q, Pointt r);
    int orientation(Pointt p, Pointt q, Pointt r);
    bool doIntersect(Pointt p1, Pointt q1, Pointt p2, Pointt q2);
    void checkTraces(Trace traces, int& UP, int& DOWN, int& LEFT, int& RIGHT);
};

#endif /* InvadeDetector_hpp */
