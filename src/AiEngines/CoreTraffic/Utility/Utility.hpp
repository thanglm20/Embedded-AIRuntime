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