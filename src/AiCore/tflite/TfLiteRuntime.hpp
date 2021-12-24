#ifndef TFLITERUNTIME_H
#define TFLITERUNTIME_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "tensorflow/lite/delegates/gpu/delegate.h"
// #include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

#include "Config.hpp"
#include "Processing.hpp"

// #define TFLITE_MINIMAL_CHECK(x)                                     \
//     if (!(x))                                                       \
//     {                                                               \
//         fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);    \
//         exit(1);                                                    \
//     } 

typedef struct tflite_interpreter
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    TfLiteDelegate *delegate;
}tflite_interpreter;

class TfLiteRuntime {
    private:
        tflite_interpreter *p;
    public:
        TfLiteRuntime();
        ~TfLiteRuntime();
        int initTfLiteNetwork(const char *model_path, std::string runtime);
        int excuteTfLiteDetector(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects);
        int excuteTfLiteYolo(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects);
        int excuteTfLiteClassifier(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects);
};

#endif