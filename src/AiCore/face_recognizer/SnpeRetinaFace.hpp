/******************************************************************************** 
 Module: SnpeRetinaFace.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#ifndef SNPERENTINAFACE_H
#define SNPERENTINAFACE_H

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

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
struct class_info
{
	double min_distance;
	int index;
};
class SnpeRetinaFace
{
    private:
 
        std::unique_ptr<zdl::SNPE::SNPE> retinaFace;
    public:
        SnpeRetinaFace(/* args */);
        ~SnpeRetinaFace(); 
        int initSnpeRetinaFace(std::string containerPath);     
        int detect(int width, int height);
};

#endif
