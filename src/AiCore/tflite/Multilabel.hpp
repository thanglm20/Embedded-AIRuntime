#include <stdio.h>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "AiTypeData.hpp"


using namespace cv;
using namespace std;
using namespace tflite;
#define TFLITE_MINIMAL_CHECK(x)                                     \
    if (!(x))                                                       \
    {                                                               \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);    \
        exit(1);                                                    \
    }
class TfMultilabel
{
    private:
        int model_width;
        int model_height;
        int model_channels;
        tflite::ops::builtin::BuiltinOpResolver resolvert;
    public:
        TfMultilabel(tflite::ops::builtin::BuiltinOpResolver& resolvert);
        ~TfMultilabel();
        std::unique_ptr<tflite::FlatBufferModel> initModel(const char* containerPath);  
        int executeModel(cv::Mat& bgr, std::unique_ptr<tflite::FlatBufferModel> &model, json& vec_out);
};
