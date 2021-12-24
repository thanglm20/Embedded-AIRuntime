#include "TfLiteRuntime.hpp"

TfLiteRuntime :: TfLiteRuntime()
{
    this->p = new tflite_interpreter;
}

TfLiteRuntime :: ~TfLiteRuntime()
{
    TfLiteGpuDelegateV2Delete(this->p->delegate);
}

float expit(float x)
{
    return 1.f / (1.f + expf(-x));
}

float iou(cv::Rect& rect1, cv::Rect& rect2) 
{
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    int w = std::max(0, (x2 - x1 + 1));
    int h = std::max(0, (y2 - y1 + 1));
    float inter = w * h;
    float area1 = rect1.width * rect1.height;
    float area2 = rect2.width * rect2.height;
    float o = inter / (area1 + area2 - inter);
    return (o >= 0) ? o : 0;
}

void nms(std::vector<ObjectTrace>& boxes, const double nms_threshold)
{
    std::vector<int> scores;
    for(int i=0; i < boxes.size(); i++) {
        scores.push_back(boxes[i].score);
    }
    std::vector<int> index;
    for(int i=0; i < scores.size(); i++) {
        index.push_back(i);
    }
    sort(index.begin(), index.end(), [&](int a, int b){return scores[a] > scores[b];});
    
    std::vector<bool> del(scores.size(), false);
    for(size_t i=0; i < index.size(); i++) {
        if(!del[index[i]]) {
            for(size_t j=i+1; j < index.size(); j++) {
                if(iou(boxes[index[i]].rect, boxes[index[j]].rect) > nms_threshold) {
                    del[index[j]] = true;
                }
            }
        }
    }
    std::vector<ObjectTrace> new_obj;
    for(const auto i : index) {
        ObjectTrace obj;
        if(!del[i]) {
            obj.label = boxes[i].label;
            obj.score = boxes[i].score;
            obj.rect.x = boxes[i].rect.x;
            obj.rect.y = boxes[i].rect.y;
            obj.rect.width = boxes[i].rect.width;
            obj.rect.height = boxes[i].rect.height;
        }
        new_obj.push_back(obj);
    }
    boxes = new_obj;
}

int TfLiteRuntime::initTfLiteNetwork(const char *model_path, std::string runtime)
{
    this->p->model = tflite::FlatBufferModel::BuildFromFile(model_path);
    tflite::InterpreterBuilder(*(this->p->model), this->p->resolver)(&(this->p->interpreter));

    if(runtime == "GPU") {
        const TfLiteGpuDelegateOptionsV2 options = {
                                                    .is_precision_loss_allowed=0,
                                                    .inference_preference=TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER, 
                                                    .inference_priority1=TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION, 
                                                    .inference_priority2=TFLITE_GPU_INFERENCE_PRIORITY_AUTO, 
                                                    .inference_priority3=TFLITE_GPU_INFERENCE_PRIORITY_AUTO, 
                                                    .experimental_flags=TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT,
                                                    .max_delegated_partitions=6,
                                                    };
        this->p->delegate = TfLiteGpuDelegateV2Create(&options);

        tflite::Interpreter::TfLiteDelegatePtr delegate_ptr(this->p->delegate, 
                                                            [](TfLiteDelegate *delegate_ptr)
                                                                {TfLiteGpuDelegateV2Delete(delegate_ptr);});
        if(this->p->interpreter->ModifyGraphWithDelegate(std::move(delegate_ptr.get())) != kTfLiteOk) {
            return -1;
        }
    } else if(runtime == "DSP") {
        // const char* library_directory_path = "/data/tflite/hexagon_nn";
        // TfLiteHexagonInitWithPath(library_directory_path);

        // const TfLiteHexagonDelegateOptions params = {
        //                                              .debug_level = 0,
        //                                              .powersave_level = 0,
        //                                              .print_graph_profile = true,
        //                                              .print_graph_debug = true,
        //                                              .max_delegated_partitions = 6,
        //                                              .min_nodes_per_partition = 1,
        //                                              .enable_dynamic_batch_size = false,
        //                                             //  .max_batch_size = 0,
        //                                             //  .input_batch_dimensions = TfLiteIntArray*,
        //                                             //  .output_batch_dimensions = TfLiteIntArray*,
        //                                             };
        // this->p->delegate = TfLiteHexagonDelegateCreate(&params);

        // tflite::Interpreter::TfLiteDelegatePtr delegate_ptr(this->p->delegate,
        //                                                     [](TfLiteDelegate *delegate_ptr)
        //                                                         {TfLiteHexagonDelegateDelete(delegate_ptr);});
        // if(this->p->interpreter->ModifyGraphWithDelegate(std::move(delegate_ptr.get())) != kTfLiteOk) {
        //     return -1;
        // }
        std::cout << "Hexagon Delegate is not supported!" << std::endl;
    } else {
        runtime = "CPU";
        this->p->interpreter->AllocateTensors();
        this->p->interpreter->SetAllowFp16PrecisionForFp32(true);
        this->p->interpreter->SetNumThreads(4);
    }

    return 0;
}

int TfLiteRuntime::excuteTfLiteDetector(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects)
{
    TfLiteTensor* output_locations = nullptr;
    TfLiteTensor* output_classes = nullptr;
    TfLiteTensor* output_detections = nullptr;

    int width = img.cols;
    int height = img.rows;

    //INT8
    // int input = this->p->interpreter->inputs()[0];
    // TfLiteIntArray* dims = this->p->interpreter->tensor(input)->dims;
    // int wanted_height = dims->data[1];
    // int wanted_width = dims->data[2];
    // int wanted_chanels = dims->data[3];
    // cv::Mat resize;
    // cv::resize(img, resize, cv::Size(wanted_width, wanted_height));
    // memcpy(this->p->interpreter->typed_input_tensor<uchar>(0), resize.data, resize.total() * resize.elemSize());
    printf("------------------------\n");
    //FLOAT32
    int input = this->p->interpreter->inputs()[0];
    float* data = this->p->interpreter->typed_tensor<float>(input);
    TfLiteIntArray* dims = this->p->interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_chanels = dims->data[3];
    cv::Mat resize;
    cv::resize(img, resize, cv::Size(wanted_width, wanted_height));
    uint8_t* in = resize.ptr<uint8_t>(0);
    printf("------------------------\n");
    ProcessInputWithFloatModel(in, data, wanted_width, wanted_height, wanted_chanels);

    // execute
    if(this->p->interpreter->Invoke() != kTfLiteOk) {
        return false;
    }
    printf("------------------------\n");
    output_locations = this->p->interpreter->tensor(this->p->interpreter->outputs()[0]);
    auto output_data = output_locations->data.f;

    output_classes = this->p->interpreter->tensor(this->p->interpreter->outputs()[1]);
    auto output_cls = output_classes->data.f;

    output_detections = this->p->interpreter->tensor(this->p->interpreter->outputs()[3]);
    auto nums = output_detections->data.f;

    std::vector<float> locations;
    std::vector<float> cls;
    for(int i=0; i < 20; i++) {
        auto output = output_data[i];
        locations.push_back(output);
        cls.push_back(output_cls[i]);
    }

    int count = 0;
    objects.clear();

    for(int j=0; j < locations.size(); j+=4) {
        auto ymin = locations[j] * height;
        auto xmin = locations[j+1] * width;
        auto ymax = locations[j+2] * height;
        auto xmax = locations[j+3] * width;
        auto out_width = xmax - xmin;
        auto out_height = ymax - ymin;

        float score = expit(nums[count]);

        if(score < 0.6f) continue;

        ObjectTrace object;
        object.label = cls[count];
        object.score = score;
        object.rect.x = xmin;
        object.rect.y = ymin;
        object.rect.width = out_width;
        object.rect.height = out_height;
        objects.push_back(object);
        count += 1;
    }
    nms(objects, 0.6);

    return 0;
}

int TfLiteRuntime::excuteTfLiteYolo(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects)
{
    TfLiteTensor* output_locations = nullptr;
    TfLiteTensor* output_scores = nullptr;

    int width = img.cols;
    int height = img.rows;

    int input = this->p->interpreter->inputs()[0];
    float* data = this->p->interpreter->typed_tensor<float>(input);

    TfLiteIntArray* dims = this->p->interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_chanels = dims->data[3];

    cv::Mat resize;
    cv::resize(img, resize, cv::Size(wanted_width, wanted_height));

    uint8_t* in = resize.ptr<uint8_t>(0);

    ProcessInputWithFloatModel(in, data, wanted_width, wanted_height, wanted_chanels);
    
    if(this->p->interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    output_locations = this->p->interpreter->tensor(this->p->interpreter->outputs()[0]);
    auto data_locations = output_locations->data.f;

    output_scores = this->p->interpreter->tensor(this->p->interpreter->outputs()[1]);
    auto data_scores = output_scores->data.f;

    std::vector<float> locations;
    std::vector<float> scores;
    int idx = 0;
    for(int i=0; i<output_scores->dims->size*4; i++) {
        locations.push_back(data_locations[i]);
        if(i % idx == 0) {
            scores.push_back(data_scores[idx]);
            idx += 1;
        }
    }

    for(int i=0; i<scores.size(); i++) {
        float x_c = locations[i*4] / wanted_width;
        float y_c = locations[i*4+1] / wanted_height;
        float w_c = locations[i*4+2] / wanted_width;
        float h_c = locations[i*4+3] / wanted_height;
        float xmin = (x_c - w_c / 2) * width;
        float ymin = (y_c - h_c / 2) * height;
        float w_out = w_c * width;
        float h_out = h_c * height;

        ObjectTrace object;
        object.label = "Unknown";
        object.rect = cv::Rect(xmin, ymin, w_out, h_out);
        object.score = scores[i];
        objects.push_back(object);
    }
    nms(objects, 0.6);

    return 0;
}

int TfLiteRuntime::excuteTfLiteClassifier(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects)
{
    int input = this->p->interpreter->inputs()[0];
    float* data = this->p->interpreter->typed_tensor<float>(input);

    TfLiteIntArray* dims = this->p->interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_chanels = dims->data[3];

    cv::Mat resize;
    cv::resize(img, resize, cv::Size(wanted_width, wanted_height));

    uint8_t* in = resize.ptr<uint8_t>(0);

    ProcessInputWithFloatModel(in, data, wanted_width, wanted_height, wanted_chanels);
    
    if(this->p->interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    const std::vector<int> outputs = this->p->interpreter->outputs();
    int output = this->p->interpreter->outputs()[0];

    TfLiteIntArray* output_dims = this->p->interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];

    float* out = this->p->interpreter->typed_output_tensor<float>(0);

    objects.clear();
    for(int i=0; i < output_size; i++) {
        objects.push_back(ObjectTrace(labels[i], (float)out[i]));
    }
    return 0;
}
