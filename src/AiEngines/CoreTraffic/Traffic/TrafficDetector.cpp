#include <iostream>
#include "TrafficDetector.hpp"
#include "ObjectDetector.hpp"


TrafficDetector:: TrafficDetector() {
    std::string namesPath = "../model/traffic/traffic.names";
    std::string cfgPath = "/media/thanglmb/AICAM/AIProject/Traffic/model/traffic/traffic-tiny.param";
    std::string weightPath = "/media/thanglmb/AICAM/AIProject/Traffic/model/traffic/traffic-tiny.bin";
    this->detector = new ObjectDetector(namesPath, cfgPath, weightPath);
    std::cout << "[INFO] Init traffic detector successfully!\n";
}

TrafficDetector::~TrafficDetector() {
    delete this->detector;
}

bool TrafficDetector::detect(const Mat& input, std::vector<bbox_t>& detected) {

    // std::vector<bbox_t> boxes;
    detected.clear();
    this->detector->processDetect(input, 0.5, detected);

    // detected.clear();
    // for (auto &box: boxes) {
    //     detected.push_back(box); 
    // }
    return true;
}