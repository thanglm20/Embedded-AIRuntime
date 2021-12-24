

#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <string>
#include <stack>
#include "ncnn/net.h"
#include "opencv2/opencv.hpp"

struct ToaDoLandmark {
    float _x;
    float _y;
};

struct bbox {
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    ToaDoLandmark point[5];
};

struct box {
    float cx;
    float cy;
    float sx;
    float sy;
};

class FaceDetector 
{
    private:
        ncnn::Net* net = nullptr;
        ncnn::UnlockedPoolAllocator* g_blob_pool_allocator_detect = nullptr;
        ncnn::PoolAllocator* g_workspace_pool_allocator_detect = nullptr;
        float _nms = 0.4;
        float _threshold = 0.6;
        float _mean_val[3] = {104.f, 117.f, 123.f};
    public:
        FaceDetector();

        int Init(const std::string &model_path);

        void nms(std::vector <bbox> &input_boxes, float NMS_THRESH);

        void Detect(cv::Mat &img, std::vector <bbox> &boxes);

        void create_anchor(std::vector <box> &anchor, int w, int h);

        static inline bool cmp(bbox a, bbox b);

        ~FaceDetector();
        
};

#endif //