#ifndef Utility_hpp
#define Utility_hpp

#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <experimental/filesystem>
#include <iomanip>
#include <fstream>
#include "ObjectDetector.hpp"
#include "ObjectTracking.hpp"

using namespace std;
using namespace cv;

// struct bbox_t {
//     unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
//     float prob;                    // confidence - probability that the object was found correctly
//     unsigned int obj_id;           // class of object - from range [0, classes-1]
//     unsigned int track_id;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
//     unsigned int frames_counter;   // counter of frames on which the object was detected
//     float x_3d, y_3d, z_3d;        // center of object (in Meters) if ZED 3D Camera is used
// };

enum vehicleType {
    OTHER = 0,
    PERSON = 1,
    XE2BANH = 3,
    XE4BANH = 13
};

vector<string> loadObjectNames(const string& filename);
unsigned int combineObject(unsigned int obj_id);
void bbox2regions(vector<bbox_t>& bboxs, regions_t& regions, vector<string>& objNames);
int convertType2Int(vector<string>& objNames, string type);
#endif /* Utility_hpp */