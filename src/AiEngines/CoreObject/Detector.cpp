

#include "Detector.hpp"

Detector::Detector()
{
    this->objectDetector = new ObjectDetector();
    if(this->objectDetector->initObjectDetector("snpe", "AnprDetect" ) != 0)
    {
        printf("Init POC BCA core object detector failed\n");
    }  
}

Detector::~Detector()
{
    this->objectDetector->~ObjectDetector();
}

bool Detector::detect(const Mat& input, vector<bbox_t>& detected)
{
    std::vector<ObjectTrace> obj_detected;
    this->objectDetector->executeObjectDetector(input, obj_detected, 0.2);
    
    detected.clear();
    for(int i = 0; i < obj_detected.size(); i++)
    {
        bbox_t object;
        object.obj_id = obj_detected[i].obj_id;
        object.prob = obj_detected[i].score;
        object.x = obj_detected[i].rect.x;
        object.y = obj_detected[i].rect.y;
        object.w = obj_detected[i].rect.width;
        object.h = obj_detected[i].rect.height;
        detected.push_back(object);
        //printf("Object detect: %d, %d, %d, %d \n",object.x, object.y, object.w, object.h );
        //cv::Rect2f rect( object.x, object.y, object.w, object.h);
        //cv::rectangle(input, obj_detected[i].rect, cv::Scalar(255, 0, 0), 1, 8);
    }
    return true;
}