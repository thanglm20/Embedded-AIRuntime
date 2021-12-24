#ifndef BlowRedlightDetector_hpp
#define BlowRedlightDetector_hpp

#include <opencv2/opencv.hpp>
#include "TrafficDetector.hpp"
#include "Utility.hpp"
#include <fstream>

struct vehicle {
    int id;
    vector<cv::Rect> box;
    int count;
    bool left = true;
    bool right = true;
    bool straight = true;
    bool isOutOfTheFrame;
};

enum LightState {
    RED = 0,
    GREEN = 1,
    YELLOW = 2,
    UNDEFINED = -1,
    OFF = 3
};

class BlowRedLightDetector {
public:
    BlowRedLightDetector();
    ~BlowRedLightDetector();

    void update(Mat& frame, std::vector<bbox_t> detected, vector<OutputData>& boxes);
    void setData(vector<vector<Point>> regionsSet, vector<string> labels, int allowRight, int allowLeft);
    void getLightLocation(Point& plateLocation);
private:
    ObjectTracking* tracking;
    vector<string> objNames;
    vector<vector<Point>> leftLight, standardLight;
    vector<Point> beginLine, endLine, leftLine, rightLine;
    bool isBeginLine, isEndLine, isLeftLine, isRightLine;
    vector<Point> fullLane, straightLane, leftLane, rightLane, straight_leftLane, straight_rightlLane;
    
    LightState standardSignal, leftSignal;

    vector<vehicle> listVehicle;
    bool allow2TurnRight, allow2TurnLeft;
    
    bool checkLane(TrackingObject track, vector<Point> area);
    bool allow2turn(int allowturn, string type);
    LightState getCurrentLightState(const Mat &input, vector<vector<Point>> light);
    bool CheckIntersection(const TrackingObject& track, float xMax, float yMax);
    string getStatus(LightState standardSignal, LightState leftSignal);
    bool checkStill(TrackingObject tracker);
    int convertType2Int(vector<string>& objNames, string type);
};

#endif // BlowRedlightDetector_hpp
