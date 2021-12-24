#ifndef ARCFACE_H
#define ARCFACE_H

#include <cmath>
#include <vector>
#include <string>
#include "ncnn/net.h"
#include <opencv2/highgui.hpp>
#include <numeric>
#include <math.h>
#include <stdio.h>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc.hpp" 
#include <opencv2/core/core.hpp>
#include <android/log.h>

using namespace std;
using namespace cv;

struct class_info
{
	double min_distance;
	int index;
};

// Mat Zscore(const Mat &fc);

inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2);

class Arcface 
{
    public:
        Arcface();
        ~Arcface();
        void Init(string model_folder);
        cv::Mat getFeature(cv::Mat img);
        class_info classify(const cv::Mat& img, const cv::Mat& cmp);

    private:
        ncnn::Net net;

        const int feature_dim = 128;

        void normalize(vector<float> &feature);
};

#endif