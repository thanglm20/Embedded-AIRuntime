
#ifndef  SNPERUNTIME_H
#define SNPERUNTIME_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
// include snpe header
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/String.hpp"
#include "Util.hpp"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include "DlSystem/PlatformConfig.hpp"

// include my header
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Util.hpp"


#define SNPE_RUNTIME zdl::DlSystem::Runtime_t::DSP
#define SNPE_FALLBACK zdl::DlSystem::Runtime_t::GPU

typedef struct snpeBuilders
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::Runtime_t runtime;
}snpeBuilders;

bool SetAdspLibraryPath(std::string nativeLibPath);

std::unique_ptr<zdl::SNPE::SNPE> setBuilderSNPE(std::string containerPath, std::vector<std::string> outputLayers, zdl::DlSystem::Runtime_t target_device);
//GPU_FLOAT16
zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime);
std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath);
std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                    zdl::DlSystem::RuntimeList runtimeList,
                                                    zdl::DlSystem::StringList outputs);
std::unique_ptr<zdl::DlSystem::ITensor> convertMat2BgrFloat(std::unique_ptr<zdl::SNPE::SNPE>& snpe,const cv::Mat& img);
std::unique_ptr<zdl::DlSystem::ITensor> convertMat2BgrFloat1(std::unique_ptr<zdl::SNPE::SNPE>& snpe);

std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensorByte(std::unique_ptr<zdl::SNPE::SNPE> snpe, std::string& fileLine);
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensorByteToFloat(std::unique_ptr<zdl::SNPE::SNPE>& snpe, std::string& fileLine);
std::unique_ptr<zdl::DlSystem::ITensor> creatTensorBGR(std::unique_ptr<zdl::SNPE::SNPE>& snpe,const uint8_t* rawData);

#endif