/******************************************************************************** 
 Module: SnpeInsightface.cpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/

#include "SnpeInsightface.hpp"
#include "SnpeCommLib.hpp"

SnpeInsightface::SnpeInsightface(/* args */)
{
    this->snpeInsightface = std::unique_ptr<zdl::SNPE::SNPE>();
}

SnpeInsightface::~SnpeInsightface()
{
    this->snpeInsightface.release();
}

int SnpeInsightface::initSnpeInsightface(std::string containerPath)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(containerPath);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file Insightface." << std::endl;
       return -1;
    }
    // add runtime target
    static zdl::DlSystem::RuntimeList runtimeList;
    zdl::DlSystem::Runtime_t target_device =  zdl::DlSystem::Runtime_t::DSP;
    static zdl::DlSystem::Runtime_t runtime = checkRuntime(target_device);
    if(runtimeList.empty()) {
        runtimeList.add(runtime);
    }    
    // adding of second layer gives us three more buffers which will have boxes and scores
    zdl::DlSystem::StringList outputs;
    
    this->snpeInsightface = setBuilderOptions(container, runtimeList, outputs);
    if (this->snpeInsightface == nullptr)
    {
       std::cerr << "Error while building SNPE object Insightface." << std::endl;
       //return nullptr;
       return -1;
    }
    return 0;
}
cv::Mat SnpeInsightface::executeInsightface(const cv::Mat& img)
{
    // Align face for recognize
    

    //Execute the network and store the outputs that were specified when creating the network in a TensorMap
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    input =  convertMat2BgrFloat(this->snpeInsightface, img);
    static zdl::DlSystem::TensorMap outputTensorMap;
    int exeStatus  = this->snpeInsightface->execute(input.get(), outputTensorMap);
    if(exeStatus == true)
    {
        printf("Execute SNPE successfully \n");
    }
    else
    {
        printf("Error while executing the network \n");
    }
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    // get class, scores
    zdl::DlSystem::ITensor *outTensor = outputTensorMap.getTensor(tensorNames.at(0));
    float* outData = reinterpret_cast<float*>(&(*outTensor->begin()));
    zdl::DlSystem::TensorShape outShape = outTensor->getShape();

    std::vector<float> feature(outShape[1]);
    feature.clear();
    for(size_t t=0; t < outShape[1]; t++) {
        auto output = static_cast<float>(outData[t]);
        // std::cout << "output: " << t << " = " << output << std::endl;
        feature.push_back(output);
    }

    cv::Mat outMat = cv::Mat(feature, true).reshape(1, 1);
    cv::normalize(outMat, outMat);

    // for(int i=0; i < outMat.rows; i++) {
    //     for(int j=0; j<outMat.cols; j++)
    //     {
    //         std::cout << outMat.at<float>(i,j) <<std::endl;
    //     }
    // }
    return outMat;
}
class_info SnpeInsightface::classify(const cv::Mat& img, const  cv::Mat& cmp)
{
	int rows = cmp.rows;
	cv::Mat broad;
	cv::repeat(img, rows, 1, broad);
	broad = broad - cmp;
	cv::pow(broad,2,broad);
	cv::reduce(broad, broad, 1, cv::REDUCE_SUM);

	double dis;
	cv::Point point;
	cv::minMaxLoc(broad, &dis, 0, &point, 0);

	return class_info{dis, point.y};
}