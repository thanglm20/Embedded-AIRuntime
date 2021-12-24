/******************************************************************************** 
 Module: SnpeMtcnn.hpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#ifndef SNPEMTCNN_H
#define SNPEMTCNN_H


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
#include "comm_lib.hpp"
#include "mtcnn.hpp"

#define WIDTH_PNET 720
#define HEIGHT_PNET 360
class SnpeMtcnn
{
    private:
    /* data */
        int widthInput;
        int heightInput;
        std::vector<scale_window> windowListPnet;
        std::vector<std::unique_ptr<zdl::SNPE::SNPE>> P_net;
        std::unique_ptr<zdl::SNPE::SNPE> R_net;
        std::unique_ptr<zdl::SNPE::SNPE> O_net;
        int initPnet(std::string pathPnet);
        int initRnet(std::string pathRnet);
        int initOnet(std::string pathOnet);
        int runPnet(std::unique_ptr<zdl::SNPE::SNPE>& pnet, std::unique_ptr<zdl::DlSystem::ITensor>& input,  scale_window& win, std::vector<face_box>&box_list);
        int runRNet(std::unique_ptr<zdl::DlSystem::ITensor>& input,  face_box& input_box, face_box& output_box);
        int runONet(std::unique_ptr<zdl::DlSystem::ITensor>& input, face_box& input_box, face_box& output_box);
    public:
        SnpeMtcnn(/* args */);
        ~SnpeMtcnn();
        int mtncnnInit(std::string pathPnet, std::string pathRnet, std::string pathOnet);
        int detect( int width, int height, std::vector<face_box>& face_list);
};


#endif