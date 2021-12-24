/******************************************************************************** 
 Module: SnpeRuntime.cpp
 Author: LE MANH THANG
 Created: 08/02/2021
 Description: 
********************************************************************************/
#include "SnpeClassifier.hpp"

SnpeClassifier::SnpeClassifier(/* args */)
{
    this->snpeClassifer = std::unique_ptr<zdl::SNPE::SNPE>();
}
SnpeClassifier::~SnpeClassifier()
{
    this->snpeClassifer.release();
}
int SnpeClassifier::initSnpeClassifier(std::string containerPath)
{
     std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(containerPath);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       return -1;
    }
    printf("Loaded dlc file successfully\n");
    // add runtime target
    static zdl::DlSystem::RuntimeList runtimeList;
    zdl::DlSystem::Runtime_t target_device =  zdl::DlSystem::Runtime_t::DSP;
    static zdl::DlSystem::Runtime_t runtime = checkRuntime(target_device);
    if(runtimeList.empty()) {
        runtimeList.add(runtime);
    }    
    // adding of second layer gives us three more buffers which will have boxes and scores
    zdl::DlSystem::StringList outputs;
    this->snpeClassifer = setBuilderOptions(container, runtimeList, outputs);
    if (this->snpeClassifer == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       //return nullptr;
       return -1;
    }
    printf("Configured SNPE Classifier network successfully\n");
    //return snpe;
    return 0;
}
int SnpeClassifier::executeSnpeClassifier(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects)
{
    //Execute the network and store the outputs that were specified when creating the network in a TensorMap
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    input =  convertMat2BgrFloat(this->snpeClassifer, img);
    static zdl::DlSystem::TensorMap outputTensorMap;
    int exeStatus  = this->snpeClassifer->execute(input.get(), outputTensorMap);
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
    float scoreMax = 0.0;
    int classID = 0;
    objects.clear();
    for (size_t j = 0; j < outTensor->getSize(); j++) 
    {
        ObjectTrace obj;
        obj.label = labels[j];
        obj.score = outData[j];
        objects.push_back(obj);
    }
    return 0;
}