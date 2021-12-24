#include "scrfd.hpp"
#include "SnpeCommLib.hpp"
#include "AiTypeData.hpp"


SnpeScrfd::SnpeScrfd()
{
    static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    std::cout << "SNPE version: " << version.asString().c_str() << std::endl;

    this->snpeScrfd = std::unique_ptr<zdl::SNPE::SNPE>();
}

SnpeScrfd::~SnpeScrfd()
{
    if(this->snpeScrfd)
        this->snpeScrfd.release();
}

static inline float intersection_area(const FaceSCRFD &a, const FaceSCRFD &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceSCRFD> &face_objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = face_objects[(left + right) / 2].prob;

    while(i <= j) {
        while(face_objects[i].prob > p)
            i++;
        
        while(face_objects[j].prob < p)
            j--;

        if(i <= j) {
            std::swap(face_objects[i], face_objects[j]);
            i++;
            j--;
        }
    }

    {
        {
            if(left < j) qsort_descent_inplace(face_objects, left, j);
        }

        {
            if(i < right) qsort_descent_inplace(face_objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceSCRFD> &face_objects)
{
    if(face_objects.empty())
        return ;
    
    qsort_descent_inplace(face_objects, 0, face_objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceSCRFD> &face_objects, std::vector<int> &picked, float threshold_nms)
{
    picked.clear();

    const int n = face_objects.size();

    std::vector<float> areas(n);
    for(int i=0; i<n; i++) {
        areas[i] = face_objects[i].rect.area();
    }

    for(int i=0; i<n; i++) {
        const FaceSCRFD &a = face_objects[i];

        int keep = 1;
        for(int j=0; j<(int)picked.size(); j++) {
            const FaceSCRFD &b = face_objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            
            if(inter_area / union_area > threshold_nms)
                keep = 0;
        }
        if(keep)
            picked.push_back(i);
    }
}

static void generate_anchors(int base_size, int feat_stride, float anchors[][2])
{
    int row = 0;
    for(int i=0; i<base_size; i++) {
        for(int j=0; j<base_size; j++) {
            anchors[row][0] = j * feat_stride;
            anchors[row][1] = i * feat_stride;
            anchors[row + 1][0] = j * feat_stride;
            anchors[row + 1][1] = i * feat_stride;
            row += 2;
        }
    }
}

static void generate_proposals(float anchors[][2], int feat_stride, const float *oScore, const float *oBbox, const float *oKps, 
                                size_t size, std::vector<FaceSCRFD> &face_objects)
{
    for(size_t i=0; i<size; i++) {
        float prob = static_cast<float>(oScore[i]);
        if(prob >= prob_threshold) {
            float x0 = anchors[i][0] - static_cast<float>(oBbox[4 * i] * feat_stride);
            float y0 = anchors[i][1] - static_cast<float>(oBbox[4 * i + 1] * feat_stride);
            float x1 = anchors[i][0] + static_cast<float>(oBbox[4 * i + 2] * feat_stride);
            float y1 = anchors[i][1] + static_cast<float>(oBbox[4 * i + 3] * feat_stride);

            x0 = std::max(std::min(x0, (float)input_width - 1), 0.f);
            y0 = std::max(std::min(y0, (float)input_height - 1), 0.f);
            x1 = std::max(std::min(x1, (float)input_width - 1), 0.f);
            y1 = std::max(std::min(y1, (float)input_height - 1), 0.f);

            float l_x0 = anchors[i][0] + static_cast<float>(oKps[10 * i] * feat_stride);
            float l_y0 = anchors[i][1] + static_cast<float>(oKps[10 * i + 1] * feat_stride);
            float l_x1 = anchors[i][0] + static_cast<float>(oKps[10 * i + 2] * feat_stride);
            float l_y1 = anchors[i][1] + static_cast<float>(oKps[10 * i + 3] * feat_stride);
            float l_x2 = anchors[i][0] + static_cast<float>(oKps[10 * i + 4] * feat_stride);
            float l_y2 = anchors[i][1] + static_cast<float>(oKps[10 * i + 5] * feat_stride);
            float l_x3 = anchors[i][0] + static_cast<float>(oKps[10 * i + 6] * feat_stride);
            float l_y3 = anchors[i][1] + static_cast<float>(oKps[10 * i + 7] * feat_stride);
            float l_x4 = anchors[i][0] + static_cast<float>(oKps[10 * i + 8] * feat_stride);
            float l_y4 = anchors[i][1] + static_cast<float>(oKps[10 * i + 9] * feat_stride);

            l_x0 = std::max(std::min(l_x0, (float)input_width - 1), 0.f);
            l_y0 = std::max(std::min(l_y0, (float)input_height - 1), 0.f);
            l_x1 = std::max(std::min(l_x1, (float)input_width - 1), 0.f);
            l_y1 = std::max(std::min(l_y1, (float)input_height - 1), 0.f);
            l_x2 = std::max(std::min(l_x2, (float)input_width - 1), 0.f);
            l_y2 = std::max(std::min(l_y2, (float)input_height - 1), 0.f);
            l_x3 = std::max(std::min(l_x3, (float)input_width - 1), 0.f);
            l_y3 = std::max(std::min(l_y3, (float)input_height - 1), 0.f);
            l_x4 = std::max(std::min(l_x4, (float)input_width - 1), 0.f);
            l_y4 = std::max(std::min(l_y4, (float)input_height - 1), 0.f);

            FaceSCRFD obj;
            obj.prob = prob;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0 + 1;
            obj.rect.height = y1 - y0 + 1;
            obj.landmark[0].x = l_x0;
            obj.landmark[0].y = l_y0;
            obj.landmark[1].x = l_x1;
            obj.landmark[1].y = l_y1;
            obj.landmark[2].x = l_x2;
            obj.landmark[2].y = l_y2;
            obj.landmark[3].x = l_x3;
            obj.landmark[3].y = l_y3;
            obj.landmark[4].x = l_x4;
            obj.landmark[4].y = l_y4;
            face_objects.push_back(obj);
        }

    }
} 

int SnpeScrfd::initSnpeScrfd(std::string containerPath, zdl::DlSystem::Runtime_t targetDevice)
{
    std::cout << "## path model: " << containerPath << std::endl;
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(containerPath.c_str());
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       return -1;
    }
    printf("Loaded dlc file successfully\n");

    if(!zdl::SNPE::SNPEFactory::isRuntimeAvailable(targetDevice)) {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        targetDevice = zdl::DlSystem::Runtime_t::CPU;
    }

    // zdl::DlSystem::TensorShapeMap inputShapeMap;
    // inputShapeMap.add("input.1", {1, 640, 360, 3});

    zdl::DlSystem::StringList outputs;
    // stride 8
    outputs.append("Sigmoid_166");
    outputs.append("Reshape_169");
    outputs.append("Reshape_172");
    // stride 16
    outputs.append("Sigmoid_145");
    outputs.append("Reshape_148");
    outputs.append("Reshape_151");
    // stride 32
    outputs.append("Sigmoid_124");
    outputs.append("Reshape_127");
    outputs.append("Reshape_130");

    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    this->snpeScrfd = snpeBuilder.setOutputLayers(outputs)
                                    .setRuntimeProcessor(targetDevice)
                                    .setCPUFallbackMode(true)
                                    .build();
    

    return 0;
}

void SnpeScrfd::executeSnpe(std::unique_ptr<zdl::DlSystem::ITensor> &input)
{
    static zdl::DlSystem::TensorMap outputTensorMap;
    int exeStatus = this->snpeScrfd->execute(input.get(), outputTensorMap);
    if(exeStatus == true) {
        std::cout << "Execute SNPE Successfully" << std::endl;
    } else {
        std::cout << "Error while executing the network!" << std::endl;
    }

    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    for(size_t i=0; i<tensorNames.size(); i++) {
        zdl::DlSystem::ITensor *outTensor_i = outputTensorMap.getTensor(tensorNames.at(i));
        zdl::DlSystem::TensorShape shape_i = outTensor_i->getShape();
        std::cout << tensorNames.at(i) << " (";
        for(size_t j=0; j<shape_i.rank(); j++) {
            std::cout << shape_i[j] << ",";
        }
        std::cout << ")" << std::endl;
    }

}
inline float divideEle(int x)
{
    return (float)(x / 128.0f - 1);
}
inline std::unique_ptr<zdl::DlSystem::ITensor> convertIT(std::unique_ptr<zdl::SNPE::SNPE>& snpe,const cv::Mat& img)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert (strList.size() == 1);
    // Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size will be represented with a value of 0. /
    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;
    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains the expected number of elements.
       With the input dimensions computed create a tensor to convey the input into the network. */
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
   // printf("SNPE input shape: [%d, %d, %d, %d]\n", inputShape[0], inputShape[1], inputShape[2], inputShape[3]);  
    //Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    
    // printf("Input network target: %d with batch %d, h %d, w %d, c %d\n", input->getSize(),inputShape[0], inputShape[1], inputShape[2], inputShape[3] );
    u_int16_t SRC_WIDTH = inputShape[1];
    u_int16_t SRC_HEIGHT = inputShape[2];
    u_int16_t SRC_DEPTH = inputShape[3];

    auto start = std::chrono::high_resolution_clock::now();    
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
    cv::Mat bgr_resize;  
    cv::resize(bgr, bgr_resize, cv::Size(SRC_WIDTH, SRC_HEIGHT), cv::INTER_LINEAR);   
    cv::Mat flat = bgr_resize.reshape(1, bgr_resize.total() * bgr_resize.channels());
    std::vector<unsigned char> inputVec = bgr_resize.isContinuous() ? flat : flat.clone();
    auto end = std::chrono::high_resolution_clock::now();    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (input->getSize() != inputVec.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << inputVec.size() << "\n";
        return nullptr;
    }
    
    std::transform( inputVec.begin(), inputVec.end(), input->begin(), divideEle );
    return input;
}
void SnpeScrfd::executeSnpeScrfd(const cv::Mat &img, std::vector<FaceSCRFD> &face_objects)
{
    // auto start = std::chrono::high_resolution_clock::now();
    auto start0 = std::chrono::high_resolution_clock::now();    

    std::unique_ptr<zdl::DlSystem::ITensor> input;
    input = convertMat2BgrFloat(this->snpeScrfd, img);
    auto end0 = std::chrono::high_resolution_clock::now();    
    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
    std::cout << "Time creation of tensor: " <<  duration0.count() << std::endl;

    
    static zdl::DlSystem::TensorMap outputTensorMap;
    int exeStatus = this->snpeScrfd->execute(input.get(), outputTensorMap);
    /*
    // auto end = std::chrono::high_resolution_clock::now();
    // auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "duration = " << time.count() << std::endl;

    if(exeStatus == true) {
        std::cout << "Execute SNPE Successfully" << std::endl;
    } else {
        std::cout << "Error while executing the network!" << std::endl;
    }

    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

    std::vector<FaceSCRFD> face_proposals;

    // stride 8
    {
        std::string scoreName = "443";
        std::string bboxName = "446";
        std::string kpsName = "449";

        zdl::DlSystem::ITensor *outTensorScore = outputTensorMap.getTensor(scoreName.c_str());
        zdl::DlSystem::ITensor *outTensorBbox = outputTensorMap.getTensor(bboxName.c_str());
        zdl::DlSystem::ITensor *outTensorKps = outputTensorMap.getTensor(kpsName.c_str());

        zdl::DlSystem::TensorShape scoreShape = outTensorScore->getShape();
        // std::cout << "scoreShape: " << " (";
        // for(size_t j=0; j<scoreShape.rank(); j++) {
        //     std::cout << scoreShape[j] << ",";
        // }
        // std::cout << ")" << std::endl;

        zdl::DlSystem::TensorShape bboxShape = outTensorBbox->getShape();
        // std::cout << "bboxShape: " << " (";
        // for(size_t j=0; j<bboxShape.rank(); j++) {
        //     std::cout << bboxShape[j] << ",";
        // }
        // std::cout << ")" << std::endl;

        zdl::DlSystem::TensorShape kpsShape = outTensorKps->getShape();
        // std::cout << "kpsShape: " << " (";
        // for(size_t j=0; j<kpsShape.rank(); j++) {
        //     std::cout << kpsShape[j] << ",";
        // }
        // std::cout << ")" << std::endl;

        const auto *oScore = reinterpret_cast<float *>(&(*outTensorScore->begin()));
        const auto *oBbox = reinterpret_cast<float *>(&(*outTensorBbox->begin()));
        const auto *oKps = reinterpret_cast<float *>(&(*outTensorKps->begin()));

        const int base_size = 80;
        const int feat_stride = 8;
        const int num_anchor = 2;
        const int cols_anchor = 2;
        const int rows_anchor = base_size * base_size * num_anchor;

        float anchors[rows_anchor][cols_anchor];
        generate_anchors(base_size, feat_stride, anchors);

        std::vector<FaceSCRFD> face_objects_32;
        generate_proposals(anchors, feat_stride, oScore, oBbox, oKps, scoreShape[0], face_objects_32);

        face_proposals.insert(face_proposals.end(), face_objects_32.begin(), face_objects_32.end());
        
    }

    // stride 16
    {
        std::string scoreName = "468";
        std::string bboxName = "471";
        std::string kpsName = "474";

        zdl::DlSystem::ITensor *outTensorScore = outputTensorMap.getTensor(scoreName.c_str());
        zdl::DlSystem::ITensor *outTensorBbox = outputTensorMap.getTensor(bboxName.c_str());
        zdl::DlSystem::ITensor *outTensorKps = outputTensorMap.getTensor(kpsName.c_str());

        zdl::DlSystem::TensorShape scoreShape = outTensorScore->getShape();
        zdl::DlSystem::TensorShape bboxShape = outTensorBbox->getShape();
        zdl::DlSystem::TensorShape kpsShape = outTensorKps->getShape();

        const auto *oScore = reinterpret_cast<float *>(&(*outTensorScore->begin()));
        const auto *oBbox = reinterpret_cast<float *>(&(*outTensorBbox->begin()));
        const auto *oKps = reinterpret_cast<float *>(&(*outTensorKps->begin()));

        const int base_size = 40;
        const int feat_stride = 16;
        const int num_anchor = 2;
        const int cols_anchor = 2;
        const int rows_anchor = base_size * base_size * num_anchor;

        float anchors[rows_anchor][cols_anchor];
        generate_anchors(base_size, feat_stride, anchors);

        std::vector<FaceSCRFD> face_objects_16;
        generate_proposals(anchors, feat_stride, oScore, oBbox, oKps, scoreShape[0], face_objects_16);

        face_proposals.insert(face_proposals.end(), face_objects_16.begin(), face_objects_16.end());
    }

    // stride 32
    {
        std::string scoreName = "493";
        std::string bboxName = "496";
        std::string kpsName = "499";

        zdl::DlSystem::ITensor *outTensorScore = outputTensorMap.getTensor(scoreName.c_str());
        zdl::DlSystem::ITensor *outTensorBbox = outputTensorMap.getTensor(bboxName.c_str());
        zdl::DlSystem::ITensor *outTensorKps = outputTensorMap.getTensor(kpsName.c_str());

        zdl::DlSystem::TensorShape scoreShape = outTensorScore->getShape();
        zdl::DlSystem::TensorShape bboxShape = outTensorBbox->getShape();
        zdl::DlSystem::TensorShape kpsShape = outTensorKps->getShape();

        const auto *oScore = reinterpret_cast<float *>(&(*outTensorScore->begin()));
        const auto *oBbox = reinterpret_cast<float *>(&(*outTensorBbox->begin()));
        const auto *oKps = reinterpret_cast<float *>(&(*outTensorKps->begin()));

        const int base_size = 20;
        const int feat_stride = 32;
        const int num_anchor = 2;
        const int cols_anchor = 2;
        const int rows_anchor = base_size * base_size * num_anchor;

        float anchors[rows_anchor][cols_anchor];
        generate_anchors(base_size, feat_stride, anchors);

        std::vector<FaceSCRFD> face_objects_8;
        generate_proposals(anchors, feat_stride, oScore, oBbox, oKps, scoreShape[0], face_objects_8);

        face_proposals.insert(face_proposals.end(), face_objects_8.begin(), face_objects_8.end());
    }

    float width = img.cols;
    float height = img.rows;

    qsort_descent_inplace(face_proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(face_proposals, picked, _nms_threshold);

    int face_count = picked.size();

    face_objects.resize(face_count);
    for(int i=0; i<face_count; i++) {
        face_objects[i] = face_proposals[picked[i]];

        float x = face_objects[i].rect.x;
        float y = face_objects[i].rect.y;
        float w = face_objects[i].rect.width;
        float h = face_objects[i].rect.height;

        face_objects[i].rect.x = x / (float)input_width * width;
        face_objects[i].rect.y = y / (float)input_height * height;
        face_objects[i].rect.width = w / (float)input_width * width;
        face_objects[i].rect.height = h / (float)input_width * height;

        float l_x0 = face_objects[i].landmark[0].x;
        float l_y0 = face_objects[i].landmark[0].y;
        float l_x1 = face_objects[i].landmark[1].x;
        float l_y1 = face_objects[i].landmark[1].y;
        float l_x2 = face_objects[i].landmark[2].x;
        float l_y2 = face_objects[i].landmark[2].y;
        float l_x3 = face_objects[i].landmark[3].x;
        float l_y3 = face_objects[i].landmark[3].y;
        float l_x4 = face_objects[i].landmark[4].x;
        float l_y4 = face_objects[i].landmark[4].y;

        face_objects[i].landmark[0].x = l_x0 / (float)input_width * width;
        face_objects[i].landmark[0].y = l_y0 / (float)input_height * height;
        face_objects[i].landmark[1].x = l_x1 / (float)input_width * width;
        face_objects[i].landmark[1].y = l_y1 / (float)input_height * height;
        face_objects[i].landmark[2].x = l_x2 / (float)input_width * width;
        face_objects[i].landmark[2].y = l_y2 / (float)input_height * height;
        face_objects[i].landmark[3].x = l_x3 / (float)input_width * width;
        face_objects[i].landmark[3].y = l_y3 / (float)input_height * height;
        face_objects[i].landmark[4].x = l_x4 / (float)input_width * width;
        face_objects[i].landmark[4].y = l_y4 / (float)input_height * height;
    }
    */

}

int SnpeScrfd::draw(cv::Mat &rgb, const std::vector<FaceSCRFD> &face_objects)
{
    for(size_t i=0; i<face_objects.size(); i++) {
        const FaceSCRFD &obj = face_objects[i];

        cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0), 2);

        cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(0, 255, 0), 2);

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if(y < 0)
            y = 0;
        if(x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(0, 255, 0), 1);
        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    return 0;
}