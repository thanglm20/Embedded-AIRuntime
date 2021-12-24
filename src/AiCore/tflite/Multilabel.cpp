#include "Multilabel.hpp"


TfMultilabel::TfMultilabel(tflite::ops::builtin::BuiltinOpResolver& resolvert)
{
    this->resolvert = resolvert;
    // this->delegate = TfLiteGpuDelegateV2Create(nullptr);
}

TfMultilabel::~TfMultilabel()
{
    // TfLiteGpuDelegateV2Delete(this->delegate);
}

const char* Labels[] {
 "male",     //0
 "baby",  //1
 "female",      //2
 "hat",      //3
 "faceMask", //4   
 "backpack",        //5
 "blue",  //6
 "black",         //7
 "red",          //8
 "green",     //9
 "brown",         //10
 "violet",         //11
 "white",       //12
 "yellow"         //13
};

std::unique_ptr<tflite::FlatBufferModel> TfMultilabel::initModel(const char* containerPath)
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(containerPath);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    return model;
}
void ProcessInputWithFloatModel(uint8_t* input, float* buffer, int wanted_input_width, int wanted_input_height, int wanted_input_chanels)
{
    for(int y=0; y < wanted_input_height; ++y) {
        float* out_row = buffer + (y * wanted_input_width * wanted_input_chanels);
        for(int x=0; x < wanted_input_width; ++x) {
            uint8_t* input_pixel = input + (y * wanted_input_width * wanted_input_chanels) + (x * wanted_input_chanels);
            float* output_pixel = out_row + (x * wanted_input_chanels);
            for(int c=0; c < wanted_input_chanels; ++c) {
                output_pixel[c] = input_pixel[c] / 255.0f;
            }
        }
    }
}
// void FindMax(int* a[],int n)
// {
//     int* Max;
//     Max = a[0];
//     for(int i=1; i<n; i++)
//     {
//         if(a[i]>Max)
//         {
//             Max = a[i]; 
//         }
//     }
// }
bool comp(float a, float b) 
{ 
    return (a < b); 
} 
int TfMultilabel::executeModel(cv::Mat& bgr, std::unique_ptr<tflite::FlatBufferModel> &model, json& vec_out)
{
    double start, time;
    vec_out.clear();
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolvert)(&interpreter);
    interpreter->AllocateTensors();
    
    int input = interpreter->inputs()[0];
    float* data = interpreter->typed_tensor<float>(input);
    const std::vector<int> inputs = interpreter->inputs();
    //std::cout << "Number of inputs: " << inputs.size() << std::endl;

    TfLiteType input_type = interpreter->tensor(input)->type;
    // assuming one input only
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_chanels = dims->data[3];

    //std::cout << wanted_height << " x " << wanted_width << " x " << wanted_chanels << std::endl;
    cv::Mat resize;
    cv::resize(bgr, resize, cv::Size(wanted_width, wanted_height));
    cv::cvtColor(resize, resize, cv::COLOR_BGR2RGB);

    uint8_t* in = resize.ptr<uint8_t>(0);
    
    //printf("Input: %d x %d x %d\n", wanted_width, wanted_height, wanted_chanels);
    ProcessInputWithFloatModel(in, data, wanted_width, wanted_height, wanted_chanels);
    
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);

    start = static_cast<double>(cv::getTickCount());
    interpreter->Invoke();
    time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();


    //std::cout << "Execute network successfully" << std::endl;
    const std::vector<int> outputs = interpreter->outputs();
    int output = interpreter->outputs()[0];
    // get output size
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    TfLiteType output_type = interpreter->tensor(output)->type;
    float* out = interpreter->typed_output_tensor<float>(0);
    //std::cout << "Output size: " << output_size << std::endl;
    //for(int i=0;i<output_size;i++)
    //{
    //    std::cout << out[i] << std::endl;
    //}
    for (int i = 0; i < 3; i++)
    {
        if(out[i] == std::max({out[0], out[1], out[2]},comp) && out[i] >= 0.5f)
        {
            // vec_out.push_back(Labels[i]);
            vec_out["gender"] = Labels[i];
            //cout <<"Gioi tinh :"<< i <<": \t "<< std::max({out[0], out[1], out[2]},comp) << "\n";
        }
        else
        {
            vec_out["gender"] = "";
        }
    }
    if (out[3] >= 0.65f)
    {
        // vec_out.push_back(Labels[3]);
        vec_out["colorhat"] = "found";
        //std::cout << "Ti le mu: " << out[3] << std::endl;
    }
    else
    {
        vec_out["colorhat"] = "";
        //std::cout << "Ti le mu: NULL" << std::endl;
    }
    /*FAKE*/
    vec_out["colorbackpack"]  = "";
    vec_out["colortrousers"]  = "";
    // if (out[4] >= 0.6f)
    // {
    //     vec_out["FaceMask"] = "found";
    //     //std::cout << "Ti le khau trang: " << out[4] << std::endl;
    // }
    // else
    // {
    //     vec_out["FaceMask"] = " ";
    //     //std::cout << "Ti le khau trang: NULL" << std::endl;
    // }
    // if (out[5] >= 0.6f)
    // {
    //     vec_out["Backpack"] = "found";
    //     //std::cout << "Ti le balo: " << out[5] << std::endl;
    // }
    // else
    // {
    //     vec_out["Backpack"] = " ";
    //     //std::cout << "Ti le balo: NULL"<< std::endl;
    // }
    for (int i = 6; i < 14; i++)
    {
        if(out[i] == std::max({out[6], out[7], out[8],out[9], out[10], out[11],out[12],out[13]},comp) && out[i] >= 0.4f)
        {
            vec_out["colorshirt"] = Labels[i];
            //cout << "Mau sac quan quan ao "<< i <<": \t "<< std::max({out[6], out[7], out[8],out[9], out[10], out[11],out[12],out[13]},comp) << std::endl;
        }
        else if(out[i] == std::max({out[6], out[7], out[8],out[9], out[10], out[11],out[12],out[13]},comp) && out[i] < 0.3f)
        {
            vec_out["colorshirt"] = "";
        }
    }
    return 0;
}


