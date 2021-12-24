/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#include "MnnFaceDetector.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

MnnFaceDetector::MnnFaceDetector(/* args */std::string pathModel)
{
    this->ultraface = new UltraFace(pathModel, 320, 240, 4, 0.65);
}

MnnFaceDetector::~MnnFaceDetector()
{
    this->ultraface->~UltraFace();
}

int MnnFaceDetector::initMnnFaceDetector(std::string pathModel)
{
    //UltraFace ultraface(pathModel, 320, 240, 4, 0.65);
    
    return 0;
}
int MnnFaceDetector::executeMnnFaceDetector(const cv::Mat& img, std::vector<cv::Mat>& faces)
{
    faces.clear();
    std::vector<FaceInfo> face_info;
    this->ultraface->detect(img, face_info);
    for (auto face : face_info) 
    {
        cv::Mat detected_face;
        //cv::Point pt1(face.x1, face.y1);
        //cv::Point pt2(face.x2, face.y2);
        //cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        cv::Rect2f rect_crop = cv::Rect2f(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
        detected_face = img(rect_crop).clone();
        faces.push_back(detected_face);
    }
    return 0;
}