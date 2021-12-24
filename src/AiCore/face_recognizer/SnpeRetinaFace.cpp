/******************************************************************************** 
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#include "SnpeRetinaFace.hpp"
#include "SnpeCommLib.hpp"


SnpeRetinaFace::SnpeRetinaFace(/* args */)
{
    this->retinaFace = std::unique_ptr<zdl::SNPE::SNPE>();
}
SnpeRetinaFace::~SnpeRetinaFace()
{
    this->retinaFace.release();
}
int SnpeRetinaFace::initSnpeRetinaFace(std::string containerPath)
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
    
    this->retinaFace = setBuilderOptions(container, runtimeList, outputs);
    if (this->retinaFace == nullptr)
    {
       std::cerr << "Error while building SNPE object Insightface." << std::endl;
       //return nullptr;
       return -1;
    }
    return 0;
}  

int SnpeRetinaFace::detect( int width, int height)
{
    double start, time;
    //std::string file_raw = "/data/snpe/face_recognition/dataset/face12x12.raw";
    //input = loadInputTensorByteToFloat(this->P_net, file_raw);
    
    //---------------------------------------------------------
    char *fileline = (char*)"/data/snpe/face_recognition/dataset/face.raw";
    unsigned char* img_raw = (unsigned char*)malloc( 720 * 360 * 3 * sizeof(unsigned char));
    //loadImage(fileline, img_raw, 720, 360, 3);

    std::unique_ptr<zdl::DlSystem::ITensor> input;
    // input = createFloatInputSNPE(this->retinaFace, img_raw, 720, 360);

    // static zdl::DlSystem::TensorMap outputTensorMap;
    // int exeStatus  = this->retinaFace->execute(input.get(), outputTensorMap);
    // if(exeStatus == false)
    // {
    //     printf("Error while executing the network RentinaFace\n");
    //     return -1;
    // }
    input.release();
    free(img_raw);
    return 0;
}