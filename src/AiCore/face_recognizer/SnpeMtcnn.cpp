/******************************************************************************** 
 Module: SnpeMtcnn.cpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#include "SnpeMtcnn.hpp"
#include "SnpeCommLib.hpp"



SnpeMtcnn::SnpeMtcnn(/* args */)
{
    this->widthInput = WIDTH_PNET;
    this->heightInput = HEIGHT_PNET;
    cal_pyramid_list(this->heightInput, this->widthInput, min_size_, factor_, this->windowListPnet);
	std::cout << "Window list size for Pnet: " << this->windowListPnet.size() << std::endl;
    for(int i = 0; i < this->windowListPnet.size(); i++)
	{
        std::unique_ptr<zdl::SNPE::SNPE> pnet = std::unique_ptr<zdl::SNPE::SNPE>(); 
        this->P_net.push_back(std::move(pnet));
	}
    this->R_net = std::unique_ptr<zdl::SNPE::SNPE>();
    this->O_net = std::unique_ptr<zdl::SNPE::SNPE>();
}
SnpeMtcnn::~SnpeMtcnn()
{
    this->R_net.release();
    this->O_net.release();
    for(int i = 0; i < this->windowListPnet.size(); i++)
	{    
        this->P_net[i].release();
	}
}

int SnpeMtcnn::initPnet(std::string pathPnet)
{
    //---------------------------------------------------------------------------------------
    //  Load model P net
    //---------------------------------------------------------------------------------------
    std::unique_ptr<zdl::DlContainer::IDlContainer> P_container = loadContainerFromFile(pathPnet);
    if (P_container == nullptr)
    {
       std::cerr << "Error while opening the container file Pnet." << std::endl;
       return -1;
    }
    // add runtime target
    static zdl::DlSystem::RuntimeList P_runtimeList;
    zdl::DlSystem::Runtime_t P_target_device =  zdl::DlSystem::Runtime_t::DSP;
    static zdl::DlSystem::Runtime_t P_runtime = checkRuntime(P_target_device);
    if(P_runtimeList.empty()) {
        P_runtimeList.add(P_runtime);
    }    
    // adding of second layer gives us three more buffers which will have boxes and scores
    zdl::DlSystem::StringList P_outputs;
    // P_outputs.append("conv4-2");
    // P_outputs.append("prob1");
    P_outputs.append("conv4-2");
    P_outputs.append("prob1");
    
	std::cout << "====Building PNet====" << std::endl;
    for(int i = 0; i < this->windowListPnet.size(); i++)
	{
		std::cout << "scale window Pnet w: " << this->windowListPnet[i].w << ", h: " << this->windowListPnet[i].h << std::endl;
		zdl::DlSystem::TensorShapeMap inputShapeMap;
        unsigned long w = this->windowListPnet[i].w;
        unsigned long h = this->windowListPnet[i].h;
        inputShapeMap.add("data", {1, w, h, 3});
        zdl::SNPE::SNPEBuilder snpeBuilder(P_container.get());
        this->P_net[i] = snpeBuilder.setOutputLayers(P_outputs)
                .setInputDimensions(inputShapeMap)
                .setCPUFallbackMode(true)
                .setRuntimeProcessorOrder(P_runtimeList)
                .build();
        if (this->P_net[i] == nullptr)
        {
            std::cerr << "Error while building SNPE object Pnet." << std::endl;
            //return nullptr;
            return -1;
        }
	}
    printf("Init Pnet successfully\n");
    return 0;
}
int SnpeMtcnn::initRnet(std::string pathRnet)
{
    //---------------------------------------------------------------------------------------
    //  Load model R net
    //---------------------------------------------------------------------------------------
    std::unique_ptr<zdl::DlContainer::IDlContainer> R_container = loadContainerFromFile(pathRnet);
    if (R_container == nullptr)
    {
       std::cerr << "Error while opening the container file  Rnet." << std::endl;
       return -1;
    }
    // add runtime target
    static zdl::DlSystem::RuntimeList R_runtimeList;
    zdl::DlSystem::Runtime_t R_target_device =  zdl::DlSystem::Runtime_t::DSP;
    static zdl::DlSystem::Runtime_t R_runtime = checkRuntime(R_target_device);
    if(R_runtimeList.empty()) {
        R_runtimeList.add(R_runtime);
    }    
    // adding of second layer gives us three more buffers which will have boxes and scores
    zdl::DlSystem::StringList R_outputs;
    R_outputs.append("conv5-2");
    R_outputs.append("prob1");
    this->R_net = setBuilderOptions(R_container, R_runtimeList, R_outputs);
    if (this->R_net == nullptr)
    {
       std::cerr << "Error while building SNPE object Rnet." << std::endl;
       //return nullptr;
       return -1;
    }
    // get input tensor
    const auto &R_strList_opt = this->R_net->getInputTensorNames();
    if (!R_strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &R_strList = *R_strList_opt;
    assert (R_strList.size() == 1);
    const auto &R_inputDims_opt = this->R_net->getInputDimensions(R_strList.at(0));
    const auto &R_inputShape = *R_inputDims_opt;
    printf("Rnet inputshape: %d, %d, %d, %d \n", R_inputShape[0], R_inputShape[1], R_inputShape[2], R_inputShape[3]);
    return 0;
}

int SnpeMtcnn::initOnet(std::string pathOnet)
{
    //---------------------------------------------------------------------------------------
    //  Load model O net
    //---------------------------------------------------------------------------------------
    std::unique_ptr<zdl::DlContainer::IDlContainer> O_container = loadContainerFromFile(pathOnet);
    if (O_container == nullptr)
    {
       std::cerr << "Error while opening the container file Onet." << std::endl;
       return -1;
    }
    // add runtime target
    static zdl::DlSystem::RuntimeList O_runtimeList;
    zdl::DlSystem::Runtime_t O_target_device =  zdl::DlSystem::Runtime_t::DSP;
    static zdl::DlSystem::Runtime_t O_runtime = checkRuntime(O_target_device);
    if(O_runtimeList.empty()) {
        O_runtimeList.add(O_runtime);
    }    
    // adding of second layer gives us three more buffers which will have boxes and scores
    zdl::DlSystem::StringList O_outputs;
    O_outputs.append("conv6-2");
    O_outputs.append("conv6-3");
    O_outputs.append("prob1");
    this->O_net = setBuilderOptions(O_container, O_runtimeList, O_outputs);
    if (this->O_net == nullptr)
    {
       std::cerr << "Error while building SNPE object Onet." << std::endl;
       //return nullptr;
       return -1;
    }
    // get input tensor
    const auto &O_strList_opt = this->O_net->getInputTensorNames();
    if (!O_strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &O_strList = *O_strList_opt;
    assert (O_strList.size() == 1);
    const auto &O_inputDims_opt = this->O_net->getInputDimensions(O_strList.at(0));
    const auto &O_inputShape = *O_inputDims_opt;
    printf("Onet inputshape: %d, %d, %d, %d \n", O_inputShape[0], O_inputShape[1], O_inputShape[2], O_inputShape[3]);
    return 0;
}

int SnpeMtcnn::mtncnnInit(std::string pathPnet, std::string pathRnet, std::string pathOnet)
{
    initPnet(pathPnet);
    initRnet(pathRnet);
    initOnet(pathOnet);
    return 0;
}

int SnpeMtcnn::runPnet(std::unique_ptr<zdl::SNPE::SNPE>& pnet, std::unique_ptr<zdl::DlSystem::ITensor>& input, scale_window& win, std::vector<face_box>& box_list)
{
    
    static zdl::DlSystem::TensorMap outputTensorMap;
    int exeStatus  = pnet->execute(input.get(), outputTensorMap);
    if(exeStatus == false)
    {
        printf("Error while executing the network \n");
        return -1;
    }
    zdl::DlSystem::StringList outNames = outputTensorMap.getTensorNames();
    // get confidence 
    int confidence_size = 1;
    std::string confidenceName = "prob1";
    zdl::DlSystem::ITensor *outConfidenceTensor = outputTensorMap.getTensor(confidenceName.c_str());
    zdl::DlSystem::TensorShape shapeConfidence = outConfidenceTensor->getShape();
    if (shapeConfidence.rank() != 4) {
        std::cerr << "Scores should have 4 axis" << std::endl;
        return EXIT_FAILURE;
    }
    //confidence_size = shapeConfidence[0] * shapeConfidence[1] * shapeConfidence[2] * shapeConfidence[3];
    confidence_size = outConfidenceTensor->getSize();
	int feature_w = shapeConfidence[1];
    int feature_h = shapeConfidence[2];
    const float *confidence = reinterpret_cast<float*>(&(*outConfidenceTensor->begin()));
 
    // get regress
    int regress_size = 1;
    zdl::DlSystem::ITensor *outRegressTensor = outputTensorMap.getTensor(outNames.at(1));
    zdl::DlSystem::TensorShape regressShape = outRegressTensor->getShape();
    //regress_size = shapeRegress[0] * shapeRegress[1] * shapeRegress[2] * shapeRegress[3];
    //regress_size = outRegressTensor->getSize();
    if (regressShape.rank() != 4) {
        std::cerr << "Regress should have 4 axis" << std::endl;
        return EXIT_FAILURE;
    }
    const float *regress = reinterpret_cast<float *>(&(*outRegressTensor->begin()));

    // get confidence 
    std::vector<face_box> candidate_boxes;
    //int thre_size = 0;
    // for( int i = 0; i < confidence_size; i++)
	// {
    //     if(confidence[i] > 0.999 )
    //     {
    //         //printf("Conf data: %f\n", confidence[i]);
    //         thre_size ++;
    //     }
		
	// }
	//printf("Confi size: %d thres_size %d\n", confidence_size , thre_size);
    generate_bounding_box(confidence, confidence_size, regress, win.scale, pnet_threshold_, feature_h, feature_w, candidate_boxes, false);
    printf("candidate_boxes size: %d \n", candidate_boxes.size());
	nms_boxes(candidate_boxes, 0.5, NMS_UNION, box_list);
    printf("box_list size: %d \n", box_list.size());
    return 0;
}
int SnpeMtcnn::runRNet(std::unique_ptr<zdl::DlSystem::ITensor>& input, face_box& input_box, face_box& output_box)
{
    static zdl::DlSystem::TensorMap outputTensorMap;
    int exeStatus  = this->R_net->execute(input.get(), outputTensorMap);
    if(exeStatus == false)
    {
        printf("Error while executing the network \n");
        return -1;
    }
    zdl::DlSystem::StringList outNames = outputTensorMap.getTensorNames();
    // get confidence 
    int confidence_size = 1;
    std::string confidenceName = "prob1";
    zdl::DlSystem::ITensor *outConfidenceTensor = outputTensorMap.getTensor(confidenceName.c_str());
    zdl::DlSystem::TensorShape shapeConfidence = outConfidenceTensor->getShape();
    if (shapeConfidence.rank() != 2) {
        std::cerr << "Confidence should have 2 axis" << std::endl;
        return EXIT_FAILURE;
    }
    //confidence_size = shapeConfidence[0] * shapeConfidence[1] * shapeConfidence[2] * shapeConfidence[3];
    confidence_size = outConfidenceTensor->getSize();
    const float *confidence = reinterpret_cast<float *>(&(*outConfidenceTensor->begin()));
 
    // get regress
    int regress_size = 1;
    std::string regressName = "conv5-2";
    zdl::DlSystem::ITensor *outRegressTensor = outputTensorMap.getTensor(regressName.c_str());
    zdl::DlSystem::TensorShape regressShape = outRegressTensor->getShape();
    //regress_size = shapeRegress[0] * shapeRegress[1] * shapeRegress[2] * shapeRegress[3];
    regress_size = outRegressTensor->getSize();

    if (regressShape.rank() != 2) {
        std::cerr << "Regress should have 2 axis" << std::endl;
        return EXIT_FAILURE;
    }
    const float *regress = reinterpret_cast<float *>(&(*outRegressTensor->begin()));

    if (*(confidence) > rnet_threshold_) {
		output_box.x0 = input_box.x0;
		output_box.y0 = input_box.y0;
		output_box.x1 = input_box.x1;
		output_box.y1 = input_box.y1;

		output_box.score = *(confidence);

		output_box.regress[0] = regress[0];
		output_box.regress[1] = regress[1];
		output_box.regress[2] = regress[2];
		output_box.regress[3] = regress[3];

	}
    return 0;
}

int SnpeMtcnn::runONet(std::unique_ptr<zdl::DlSystem::ITensor>& input, face_box& input_box, face_box& output_box)
{
    static zdl::DlSystem::TensorMap outputTensorMap;
    int exeStatus  = this->O_net->execute(input.get(), outputTensorMap);
    if(exeStatus == false)
    {
        printf("Error while executing the network \n");
        return -1;
    }
    zdl::DlSystem::StringList outNames = outputTensorMap.getTensorNames();
    // O_outputs.append("conv6-2");
    // O_outputs.append("conv6-3");
    // O_outputs.append("prob1");

    // get confidence 
    int confidence_size = 1;
    std::string confidenceName = "prob1";
    zdl::DlSystem::ITensor *outConfidenceTensor = outputTensorMap.getTensor(confidenceName.c_str());
    zdl::DlSystem::TensorShape shapeConfidence = outConfidenceTensor->getShape();
    // if (shapeConfidence.rank() != 2) {
    //     std::cerr << "Confidence should have 2 axis" << std::endl;
    //     return EXIT_FAILURE;
    // }
    //confidence_size = shapeConfidence[0] * shapeConfidence[1] * shapeConfidence[2] * shapeConfidence[3];
    confidence_size = outConfidenceTensor->getSize();
    const float *confidence_data = reinterpret_cast<float *>(&(*outConfidenceTensor->begin()));
 
    // get regress
    int regress_size = 1;
    std::string regressName = "conv6-3";
    zdl::DlSystem::ITensor *outRegressTensor = outputTensorMap.getTensor(regressName.c_str());
    zdl::DlSystem::TensorShape regressShape = outRegressTensor->getShape();
    //regress_size = shapeRegress[0] * shapeRegress[1] * shapeRegress[2] * shapeRegress[3];
    regress_size = outRegressTensor->getSize();
    const float *reg_data = reinterpret_cast<float *>(&(*outRegressTensor->begin()));


    // get points
    int points_size = 1;
    std::string pointName = "conv6-3";
    zdl::DlSystem::ITensor *outPointTensor = outputTensorMap.getTensor(pointName.c_str());
    zdl::DlSystem::TensorShape pointShape = outPointTensor->getShape();
    //regress_size = shapeRegress[0] * shapeRegress[1] * shapeRegress[2] * shapeRegress[3];
    points_size = outPointTensor->getSize();
    const float *points_data = reinterpret_cast<float *>(&(*outPointTensor->begin()));
    if (*(confidence_data) > onet_threshold_) {

		output_box.x0 = input_box.x0;
		output_box.y0 = input_box.y0;
		output_box.x1 = input_box.x1;
		output_box.y1 = input_box.y1;

		output_box.score = *(confidence_data);

		output_box.regress[0] = reg_data[0];
		output_box.regress[1] = reg_data[1];
		output_box.regress[2] = reg_data[2];
		output_box.regress[3] = reg_data[3];


		for (int j = 0; j<5; j++) {
			output_box.landmark.x[j] = *(points_data + j);
			output_box.landmark.y[j] = *(points_data + j + 5);
		}

	}
    return 0;
}
int SnpeMtcnn::detect( int width, int height, std::vector<face_box>& face_list)
{
    double start, time;
    //std::string file_raw = "/data/snpe/face_recognition/dataset/face12x12.raw";
    //input = loadInputTensorByteToFloat(this->P_net, file_raw);
    
    //---------------------------------------------------------
    // char *fileline = (char*)"/data/snpe/face_recognition/dataset/face.raw";
    // unsigned char* img_raw = (unsigned char*)malloc( this->widthInput * this->heightInput * 3 * sizeof(unsigned char));
    // loadImage(fileline, img_raw,  this->widthInput, this->heightInput, 3);
    cv::Mat img = cv::imread("/data/snpe/face_recognition/dataset/face.png");
    printf("Load file successfully\n");
    
    std::vector<face_box> total_pnet_boxes;
	std::vector<face_box> total_rnet_boxes;
	std::vector<face_box> total_onet_boxes;
    start = static_cast<double>(cv::getTickCount());
    // stage 1: Run Pnet
    for(int i = 0; i < this->windowListPnet.size(); i++)
	{
		std::vector<face_box>boxes;
        std::unique_ptr<zdl::DlSystem::ITensor> inputPnet;
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
        inputPnet = convertMat2BgrFloat(this->P_net[i], bgr);
        //inputPnet = createFloatInputSNPE(this->P_net[i], img_raw, this->widthInput, this->heightInput);
		runPnet(this->P_net[i], inputPnet, this->windowListPnet[i], boxes);
        total_pnet_boxes.insert(total_pnet_boxes.end(),boxes.begin(),boxes.end());
	}
    printf("Size Pnet: %d\n", total_pnet_boxes.size());
    
    // Stage 2: Run Rnet
    std::vector<face_box> pnet_boxes;
	process_boxes(total_pnet_boxes, this->heightInput, this->widthInput, pnet_boxes);

	if (pnet_boxes.size() == 0)
		return -1;
    
    //rnet_batch_bound_
    if (pnet_boxes.size() > 10000)
	{
        // TO DO
		//RunRNet(img, pnet_boxes, total_rnet_boxes);
        printf("[INFO] - The Pnet boxes more than 10000.\n");
        return -1;
	}
	else
	{
        std::unique_ptr<zdl::DlSystem::ITensor> inputRnet;
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
        inputRnet =  convertMat2BgrFloat(this->R_net, bgr);
        printf("Size boxes Pnet: %d\n", pnet_boxes.size());
		for (unsigned int i = 0; i < pnet_boxes.size();i++)
		{
			face_box out_box;
            
			if (runRNet(inputRnet, pnet_boxes[i], out_box) < 0)
				continue;
			total_rnet_boxes.push_back(out_box);
		}
	}

    // Stage 3: Run Onet
    std::vector<face_box> rnet_boxes;
	process_boxes(total_rnet_boxes, this->heightInput, this->widthInput, rnet_boxes);

	if (rnet_boxes.size() == 0)
		return -1;

	if (rnet_boxes.size() > 10000)
	{
		printf("[INFO] - The Rnet boxes more than 10000.\n");
        return -1;
	}
	else
	{
		for (unsigned int i = 0;i<rnet_boxes.size();i++)
		{
			face_box out_box;
            std::unique_ptr<zdl::DlSystem::ITensor> inputOnet;
            cv::Mat bgr;
            cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
            inputOnet =  convertMat2BgrFloat(this->O_net, bgr);
			if (runONet(inputOnet, rnet_boxes[i], out_box)<0)
				continue;
			total_onet_boxes.push_back(out_box);
		}
	}

	//calculate the landmark
	cal_landmark(total_onet_boxes);

	//Get Final Result
	regress_boxes(total_onet_boxes);
	nms_boxes(total_onet_boxes, 0.7, NMS_MIN, face_list);
    time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
    printf("Excuting MTCNN spent time: %f\n", time);
   
    
    return 0;
}
