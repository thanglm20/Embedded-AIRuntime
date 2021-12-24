/******************************************************************************** 
 Module: SnpeRuntime.cpp
 Author: LE MANH THANG
 Created: 21/12/2020
 Description: 
********************************************************************************/

#include "SnpeCommLib.hpp"

bool SetAdspLibraryPath(std::string nativeLibPath) {
    std::stringstream path;
    path << nativeLibPath << ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    return setenv("ADSP_LIBRARY_PATH", path.str().c_str(), 1 /*override*/) == 0;
}

zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime)
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

    std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; //Print Version number

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
    {
        std::cerr << "Selected runtime not present. Falling back to GPU." << std::endl;
        runtime = SNPE_FALLBACK;
    }
    return runtime;
}
std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(containerPath.c_str()));
    return container;
}
// std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
//                                                    zdl::DlSystem::RuntimeList runtimeList,
//                                                    zdl::DlSystem::StringList outputs)
// {

//     zdl::DlSystem::Runtime_t target_device =  SNPE_RUNTIME;
//     std::unique_ptr<zdl::SNPE::SNPE> snpe;
//     zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

//     snpe = snpeBuilder.setOutputLayers(outputs)
//             .setRuntimeProcessor(runtime)
//             .setCPUFallbackMode(true)
//             .build();
//     return snpe;
// }
std::unique_ptr<zdl::SNPE::SNPE> setBuilderSNPE(std::string containerPath, std::vector<std::string> outputLayers, zdl::DlSystem::Runtime_t target_device)
{
    // load model DLC file, which is built by SNPE.
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(containerPath.c_str()));
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       return nullptr;
    }

    // choose and check target runtime
    static zdl::DlSystem::Runtime_t runtime = checkRuntime(target_device);
    if(runtime == zdl::DlSystem::Runtime_t::DSP)
        printf("===========> DSP Runtime <==============\n");
    else if(runtime == zdl::DlSystem::Runtime_t::GPU_FLOAT16 || runtime == zdl::DlSystem::Runtime_t::GPU)
        printf("===========> GPU Runtime <==============\n");
    else
        printf("===========> CPU Runtime <==============\n");

    // building network
    zdl::DlSystem::StringList outputs;
    if(outputLayers.size() > 0)
    {
        for(int i = 0; i < outputLayers.size(); i++)
            outputs.append(outputLayers[i].c_str());
    }
    zdl::DlSystem::ExecutionPriorityHint_t priority = zdl::DlSystem::ExecutionPriorityHint_t::HIGH;
    zdl::DlSystem::PerformanceProfile_t performanceProfile =  zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE; // SYSTEM_SETTINGS, BURST
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputLayers(outputs)
            .setRuntimeProcessor(runtime)
            .setCPUFallbackMode(true)
            .setExecutionPriorityHint(priority)
            .setPerformanceProfile(performanceProfile)
            .build();
    return snpe;   
}

std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensorByte(std::unique_ptr<zdl::SNPE::SNPE> snpe, std::string& fileLine)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert (strList.size() == 1);
    /* Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size will be represented with a value of 0. */
    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;
    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains the expected number of elements.
       With the input dimensions computed create a tensor to convey the input into the network. */
    //std::cout << "Input shape" <<  inputShape.getDimensions() << "\n";
    std::cout << "Input shape" <<  snpe->getInputDimensions() << "\n";
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    //Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    //printf("Input network target: %d with batch %d, w %d, h %d, c %d\n", input->getSize(),inputShape[0], inputShape[1], inputShape[2], inputShape[3] );

    // If the network has a single input, each line represents the input file to be loaded for that input
    //std::vector<float> inputVec;
    std::vector< unsigned char> inputVec;
    std::vector<unsigned char> loadedFile = loadByteDataFile(fileLine);
    inputVec.insert(inputVec.end(), loadedFile.begin(), loadedFile.end());

    if (input->getSize() != inputVec.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << inputVec.size() << "\n";
        return nullptr;
    }
    /* Copy the oaded input file contents into the networks input tensor. SNPE's ITensor supports C++ STL functions like std::copy() */
    
    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}

std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensorByteToFloat(std::unique_ptr<zdl::SNPE::SNPE>& snpe, std::string& fileLine)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    // Make sure the network requires only a single input
    assert (strList.size() == 1);
    /* Create an input tensor that is correctly sized to hold the input of the network. Dimensions that have no fixed size will be represented with a value of 0. */
    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;
    /* Calculate the total number of elements that can be stored in the tensor so that we can check that the input contains the expected number of elements.
       With the input dimensions computed create a tensor to convey the input into the network. */
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
    //Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    //printf("Input network target: %d with batch %d, w %d, h %d, c %d\n", input->getSize(),inputShape[0], inputShape[1], inputShape[2], inputShape[3] );

    // If the network has a single input, each line represents the input file to be loaded for that input
    //std::vector<float> inputVec;
    std::vector< unsigned char> inputVec;
    std::vector<unsigned char> loadedFile = loadByteDataFile(fileLine);
    inputVec.insert(inputVec.end(), loadedFile.begin(), loadedFile.end());

    if (input->getSize() != inputVec.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << inputVec.size() << "\n";
        return nullptr;
    }
    
    std::vector< float> inputVecFloat;
    for( int i = 0; i < inputVec.size(); i++)
    {        
        //float fData = (float)((inputVec[i] - 128.0f) / 128.0f);
        float fData = (float)((inputVec[i] ) / 255.0f);
        inputVecFloat.push_back(fData);
        //printf("%d\n", inputVec[i]);
    }
    /* Copy the oaded input file contents into the networks input tensor. SNPE's ITensor supports C++ STL functions like std::copy() */
    std::copy(inputVecFloat.begin(), inputVecFloat.end(), input->begin());
    return input;
}
inline float divideElement(int x)
{
    return (float)(x / 128.0f - 1);
}
std::unique_ptr<zdl::DlSystem::ITensor> convertMat2BgrFloat(std::unique_ptr<zdl::SNPE::SNPE>& snpe,const cv::Mat& img)
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

    // auto start = std::chrono::high_resolution_clock::now();    
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
    cv::Mat bgr_resize;
    cv::resize(bgr, bgr_resize, cv::Size(SRC_WIDTH, SRC_HEIGHT), cv::INTER_LINEAR);   
    cv::Mat flat = bgr_resize.reshape(1, bgr_resize.total() * bgr_resize.channels());
    std::vector<unsigned char> inputVec = bgr_resize.isContinuous() ? flat : flat.clone();
    auto end = std::chrono::high_resolution_clock::now();    
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (input->getSize() != inputVec.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << inputVec.size() << "\n";
        return nullptr;
    }
    
    std::transform( inputVec.begin(), inputVec.end(), input->begin(), divideElement );
    return input;
}


std::unique_ptr<zdl::DlSystem::ITensor> convertMat2BgrFloat1(std::unique_ptr<zdl::SNPE::SNPE>& snpe)
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

    // auto start = std::chrono::high_resolution_clock::now();    
    // cv::Mat bgr;
    // cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
    // cv::Mat bgr_resize;  
    // cv::resize(bgr, bgr_resize, cv::Size(SRC_WIDTH, SRC_HEIGHT), cv::INTER_LINEAR);   
    // cv::Mat flat = bgr_resize.reshape(1, bgr_resize.total() * bgr_resize.channels());
    // std::vector<unsigned char> inputVec = bgr_resize.isContinuous() ? flat : flat.clone();
    // auto end = std::chrono::high_resolution_clock::now();    
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::vector<unsigned char> inputVec;
    // std::vector<unsigned char> inputVec(SRC_DEPTH * SRC_HEIGHT * SRC_WIDTH);
    // for(int i = 0; i < SRC_DEPTH * SRC_HEIGHT * SRC_WIDTH; i++)
    // {
    //     inputVec[i] = 128;
    // }


    if (input->getSize() != inputVec.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << inputVec.size() << "\n";
        return nullptr;
    }
    
    std::transform( inputVec.begin(), inputVec.end(), input->begin(), divideElement );
    return input;
}


std::unique_ptr<zdl::DlSystem::ITensor> creatTensorBGR(std::unique_ptr<zdl::SNPE::SNPE>& snpe, const uint8_t* rawData)
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

    // uint32_t* __restrict dstData = new uint32_t[SRC_WIDTH * SRC_HEIGHT * SRC_DEPTH];
    // fcvColorYUV420toRGB8888u8(rawData, SRC_WIDTH, SRC_HEIGHT, (uint32_t*)dstData);
    // std::vector<uchar> inputVec;
    // inputVec.insert(inputVec.begin(), dstData, dstData + SRC_WIDTH * SRC_HEIGHT * SRC_DEPTH);
    std::vector<uchar> inputVec;
    inputVec.insert(inputVec.begin(), rawData, rawData + SRC_WIDTH * SRC_HEIGHT * SRC_DEPTH);   

    std::transform( inputVec.begin(), inputVec.end(), input->begin(), divideElement );
    // delete[] dstData;
    return input;
}
