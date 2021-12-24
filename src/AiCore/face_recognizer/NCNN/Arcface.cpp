#include "Arcface.hpp"

ncnn::Mat bgr2rgb(ncnn::Mat src)
{
    int src_w = src.w;
    int src_h = src.h;
    unsigned char* u_rgb = new unsigned char[src_w * src_h * 3];
    src.to_pixels(u_rgb, ncnn::Mat::PIXEL_BGR2RGB);
    ncnn::Mat dst = ncnn::Mat::from_pixels(u_rgb, ncnn::Mat::PIXEL_RGB, src_w, src_h);
    delete[] u_rgb;
    return dst;
}

Arcface::Arcface()
{

}

Arcface::~Arcface()
{
    this->net.clear();
}
void Arcface::Init(string model_folder)
{
    string param_file = model_folder +"/mobilefacenet.param";
    string bin_file = model_folder +"/mobilefacenet.bin";

    this->net.load_param(param_file.c_str());
    this->net.load_model(bin_file.c_str());
}
cv::Mat Arcface::getFeature(cv::Mat img)
{
    vector<float> feature;
    //cv to NCNN
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    //in = bgr2rgb(in);
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);
    feature.resize(this->feature_dim);
    for (int i = 0; i < this->feature_dim; i++)
        feature[i] = out[i];
    //normalize(feature);
    cv::Mat feature__=cv::Mat(feature,true);
    return feature__;
}

void Arcface::normalize(vector<float> &feature)
{
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++)
        sum += (float)*it * (float)*it;
    sum = sqrt(sum);
    for (auto it = feature.begin(); it != feature.end(); it++)
        *it /= sum;
}

// Mat Zscore(const Mat &fc) 
// {
//     Mat mean, std;
//     meanStdDev(fc, mean, std);
//     //cout <<"mean is :"<< mean <<"std is :"<< std << endl;
//     Mat fc_norm = (fc - mean) / std;
//     return fc_norm;
// }

inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2) 
{
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}

class_info Arcface::classify(const cv::Mat& img,const cv::Mat& cmp)
{
    vector<double> score_;
    for (unsigned int compare_ = 0; compare_ < cmp.rows; ++ compare_)
    {
        cv::Mat imgTemp;
        for(int j=0; j<cmp.cols; j++)
        {
             imgTemp.push_back(cmp.at<float>(compare_,j));
        }
        score_.push_back(CosineDistance(imgTemp, img));
    }
    int maxPosition = max_element(score_.begin(),score_.end()) - score_.begin(); 
    double score = score_[maxPosition];
    // score_.clear();
    return class_info{score, maxPosition};
}

