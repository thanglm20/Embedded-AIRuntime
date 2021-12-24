#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "Oppose.hpp"

using namespace std;
using namespace cv;
// Create ROI on Screen
struct SelectedRoi {
  // on off
  int init = 0;

  //initial coordination based on EVENT_LBUTTONDOWN
  int initX;
  int initY;

  // actual coordination 
  int actualX;
  int actualY;

  int lastX;
  int lastY;

  //Selected Rect
  Rect roiRect; 

  //Selected Mat roi
  Mat takenRoi;
  vector<Point> Line = vector<Point>(2); 
}SelectedRoi;



static void CallBackF(int event, int x, int y, int flags, void* img) {
//Mouse Right button down
  if (event == EVENT_RBUTTONDOWN) {
    // cout << "right button " << endl;
    return;
  }
//Mouse Left button down
  if (event == EVENT_LBUTTONDOWN  && SelectedRoi.init == 0) {
    SelectedRoi.initX = x;
    SelectedRoi.initY = y;
    SelectedRoi.init = 1;
    // cout << "left button DOWN" << endl; 
    return;
  }
//Mouse Left button up
  if (event == EVENT_LBUTTONUP  && SelectedRoi.init == 1) {
    SelectedRoi.lastX = x;
    SelectedRoi.lastY = y;
    SelectedRoi.init = 0;
	SelectedRoi.Line = vector<Point>{Point(SelectedRoi.initX, SelectedRoi.initY), Point(SelectedRoi.lastX,  SelectedRoi.lastY)};
    // cout << "left button UP" << endl;
    return;
  }
//Mouse move coordinates update
  if (event == EVENT_MOUSEMOVE  && SelectedRoi.init == 1) {
  
    // cout << "event mouse move"<< endl; 
    SelectedRoi.actualX = x;
    SelectedRoi.actualY = y;
    SelectedRoi.roiRect = Rect(SelectedRoi.initX, SelectedRoi.initY,
    						SelectedRoi.actualX,  SelectedRoi.actualY);
	SelectedRoi.Line = vector<Point>{Point(SelectedRoi.initX, SelectedRoi.initY), Point(SelectedRoi.actualX,  SelectedRoi.actualY)};
    return;
  }
}
void test()
{	
	snpeBuilders* snpeMobilenetSSD = new snpeBuilders;
	std::vector<std::string> outputLayers;
    snpeMobilenetSSD->runtime = zdl::DlSystem::Runtime_t::CPU;

    
    snpeMobilenetSSD->snpe = setBuilderSNPE("/media/thanglmb/Bkav/AICAM/AIProject/AiCorex86_64/models/ocr.dlc", outputLayers, snpeMobilenetSSD->runtime);
    if (snpeMobilenetSSD->snpe  == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       //return nullptr;
       return ;
    }
	cv::VideoCapture cap;
	// cap.open("/media/thanglmb/Bkav/AICAM/Testing/videos/short.mp4");
	cap.open("/media/thanglmb/Bkav/AICAM/Testing/videos/nguoc_chieu.mp4");
	if(!cap.isOpened())
	{
		printf("open video error\n");
		return ;
	}
	cv::Mat frame;
	while(1)
	{
		cap >> frame;
		if (frame.empty()) 
            break;
		
		// TO DO
		auto start = std::chrono::high_resolution_clock::now(); 
		std::vector<ObjectTrace> detected;

		//detector->executeObjectDetector(frame, detected, 0.5);
		std::unique_ptr<zdl::DlSystem::ITensor> input;
        input = convertMat2BgrFloat(snpeMobilenetSSD->snpe, frame);
        zdl::DlSystem::TensorMap outputTensorMap;
        int exeStatus  = snpeMobilenetSSD->snpe->execute(input.get(), outputTensorMap);
        if(exeStatus != true)
        {
            printf("Error while executing the MobilenetSSD network \n");
            return ;
        }
        auto end = std::chrono::high_resolution_clock::now();    
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Time detect: " <<  duration.count() << endl;

		// show result
		//resize(frame, frame, Size(1280, 720));
		char key = waitKey(1);
        imshow("Video", frame);
        if(key == 'q') 
        {
            printf("Quited\n");
            break;
        }
        if(key == 'p') 
        {
            printf("Paused\n");
            while(waitKey(1) != 'p');
        }

	}
}
int main()
{
	cv::Mat frame;
	Oppose* oppose = new Oppose();
	//vector<Point> startLine{cv::Point(800, 100), cv::Point(1280, 50)};
	// vector<Point> startLine{cv::Point(700, 100), cv::Point(1280, 100)};
	//vector<Point> endLine{cv::Point(500, 720), cv::Point(1500, 800)};
	settingsOppose settings;
	//settings.startLine = startLine;
	//settings.endLine = endLine;
	settings.allowedObjects = vector<string>{""};
	if(oppose->init(settings) != STATUS_SUCCESS) return -1;
	cv::VideoCapture cap;
	// cap.open("/media/thanglmb/Bkav/AICAM/Testing/videos/short.mp4");
	cap.open("/home/thanglmb/Downloads/giaothong_cropped.mp4");
	if(!cap.isOpened())
	{
		printf("open video error\n");
		return 0 ;
	}
	while(1)
	{
		cap >> frame;
		if (frame.empty()) 
            break;
		
		// processing
		if(oppose->set(settings) != STATUS_SUCCESS) return -1;
		vector<outDataOppose> outData;

    auto start = std::chrono::high_resolution_clock::now();    
		oppose->update(frame, outData);
    auto end = std::chrono::high_resolution_clock::now();    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // cout << "Time: " <<  duration.count() << endl;
		for(auto out : outData)
		{
			if(out.isTentative)
			{
				rectangle(frame, out.rect, Scalar(0, 255, 255), 5, 8);
				cv::putText(frame, "TENTATIVE", cv::Point(out.rect.x, out.rect.y), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 1);
			}	
			if(out.isNewEvent)
			{
				rectangle(frame, out.rect, Scalar(0, 0, 255), 5, 8);
				cv::putText(frame, "OPPOSED", cv::Point(out.rect.x, out.rect.y), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
			}
		}
		char key = waitKey(1);
		if(key == '1') 
        {
            printf("Chose roi 1!\n");
            SelectedRoi.init = 0;
            SelectedRoi.initX = 0;
            SelectedRoi.initY = 0;
            SelectedRoi.actualX = 0;
            SelectedRoi.actualY = 0;
            SelectedRoi.lastX = 0;
            SelectedRoi.lastY = 0;
            while(waitKey(1) != '1')
            {
                cv::setMouseCallback("Video", CallBackF, 0); 
				line(frame, SelectedRoi.Line.at(0), SelectedRoi.Line.at(1), Scalar( 0, 255, 255), 3, LINE_AA);
				settings.startLine = SelectedRoi.Line;
                imshow("Video", frame);
            }
        }

		if(key == '2') 
        {
            printf("Chose roi 1!\n");
            SelectedRoi.init = 0;
            SelectedRoi.initX = 0;
            SelectedRoi.initY = 0;
            SelectedRoi.actualX = 0;
            SelectedRoi.actualY = 0;
            SelectedRoi.lastX = 0;
            SelectedRoi.lastY = 0;
            while(waitKey(1) != '2')
            {
                cv::setMouseCallback("Video", CallBackF, 0); 
				line(frame, SelectedRoi.Line.at(0), SelectedRoi.Line.at(1), Scalar( 0, 255, 255), 3, LINE_AA);
				settings.endLine = SelectedRoi.Line;
                imshow("Video", frame);
            }
        }
        
        

		// show result
		//resize(frame, frame, Size(1280, 720));
        imshow("Video", frame);
        if(key == 'q') 
        {
            printf("Quited\n");
            break;
        }
        if(key == 'p') 
        {
            printf("Paused\n");
            while(waitKey(1) != 'p');
        }


	}
	return 0;
}
