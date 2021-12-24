#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include <iostream>
#include <vector>

#include "Intrusion.hpp"
using namespace std;
using namespace cv;

int main()
{
	cv::Mat frame;

	Intrusion* intrusion = new Intrusion();
	settingsIntrusion settings;
    settings.arRegionsSet = std::vector<cv::Point>{cv::Point(800, 200), cv::Point(1920, 200), cv::Point(1920, 600), cv::Point(800, 600)};
	settings.arListObjects = std::vector<std::string>{"motor","car"};
    settings.cTimeOut = 5;
	settings.cTimeRepeat = 2;
	settings.cObjectCounter = 2;
	if(intrusion->init(settings) != STATUS_SUCCESS) return -1;
	cv::VideoCapture cap;
	cap.open("/media/thanglmb/Bkav/AICAM/Testing/videos/nguoc_chieu.mp4");
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
		
		//processing
        
		vector<outDataIntrusion> outData;
        auto start = std::chrono::high_resolution_clock::now();    
		intrusion->update(frame, outData);
        auto end = std::chrono::high_resolution_clock::now();    
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		cout << "Time: " << duration.count() << endl;
        for(auto out : outData)
		{
			rectangle(frame, out.rect, Scalar(0, 0, 255), 5, 8);
			
			cv::putText(frame, "INSIDED: " + out.color , cv::Point(out.rect.x, out.rect.y), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
		}

		// show result
		resize(frame, frame, Size(1280, 720));


        imshow("Video", frame);
        char key = waitKey(1);
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



// int main()
// {
//     int w = 300;
//     int h = 300;
//     int c = 3;
//     char* file_name = "/media/thanglmb/Bkav/AICAM/AI_Models/snpe/ObjectDetection/scripts/anpr_300.raw";
//     FILE* pfile;
//     uint8_t* data;
//     pfile = fopen(file_name, "rb");
//     data = (uint8_t*)malloc(w * h * c * sizeof(uint8_t));
//     fread(data, sizeof(uint8_t), w * h * c, pfile);
//     fclose(pfile);
 
//     cv::Mat mat = cv::Mat(w, h, CV_8UC3, data);
//     while(1)
//     {
//          imshow("mat", mat);
//         char key = waitKey(1);
//         if(key == 'q') 
//         {
//             printf("Quited\n");
//             break;
//         }
//     }   
//     return 0;
// }