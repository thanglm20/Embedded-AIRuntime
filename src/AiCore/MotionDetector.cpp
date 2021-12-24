/*
* To compile: g++ opencv_test2.cpp -o opencv_test2 $(pkg-config --cflags --libs opencv)
*/
#include <MotionDetector.hpp>
#include <chrono>
MotionDetector::MotionDetector()
{
}
MotionDetector::~MotionDetector()
{
}
int MotionDetector::isMotionDetected(Mat& frame, Rect& out)
{
    int fl_motion = 0;
    Mat gray,frameDelta, thresh, curFrame;
    vector<vector<Point> > cnts;
    vector<int> ArrayX;
    vector<int> ArrayY;
    vector<int> ArrayXmax;
    vector<int> ArrayYmax;
    Rect box;
    ArrayY.clear();
    ArrayX.clear();
    ArrayYmax.clear();
    ArrayXmax.clear();
    cnts.clear();
    if(frame.empty()){
        cout << "Frame is empty" << endl;
        return -1;
    }
    frame.copyTo(curFrame);
    if(!this->lastFrame.empty())
    {    
        auto start = std::chrono::high_resolution_clock::now();    
        // cv::setUseOptimized(true);
        // cv::setNumThreads(cv::getNumThreads());
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);
        
        absdiff(this->lastFrame, gray, frameDelta);
        threshold(frameDelta, thresh, 27, 255, THRESH_BINARY);

        dilate(thresh, thresh, Mat(), Point(-1,-1), 2);
        findContours(thresh, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        auto endMotion = std::chrono::high_resolution_clock::now();  
        auto timeMotion = std::chrono::duration_cast<std::chrono::milliseconds>(endMotion - start);
        // cout << "Time in motion: " << timeMotion.count() << "ms" << " = " << 1000 / timeMotion.count() << "FPS\n";
        if(cnts.size() > 0)
        {
            for(int i = 0; i < cnts.size(); i++)
            {
                if(contourArea(cnts[i]) < THRES_MOTION) continue;
                box = boundingRect(cnts[i]);  // box chua (x,y,width,height)
                if(box.x >= 0 && box.y >= 0 && 
                    (box.x + box.width) <= frame.cols && 
                    (box.y + box.height <= frame.rows))
                    {
                        ArrayX.push_back(box.x);
                        ArrayY.push_back(box.y);
                        ArrayXmax.push_back(box.x + box.width);
                        ArrayYmax.push_back(box.y + box.height);
                    }                   
            }
            if(ArrayXmax.size() > 0 && ArrayYmax.size() > 0 && ArrayX.size() > 0 && ArrayY.size() > 0)
            {
                int x_max = *max_element(ArrayXmax.begin(), ArrayXmax.end());
                int y_max = *max_element(ArrayYmax.begin(), ArrayYmax.end());
                int x_min = *min_element(ArrayX.begin(), ArrayX.end());
                int y_min = *min_element(ArrayY.begin(), ArrayY.end());

                int x = x_min;
                int y = y_min;
                int w = x_max - x_min;
                int h = y_max - y_min;
                Rect rect = Rect(x, y, w, h); 
                out = rect;
                fl_motion = 1;
            }
        }
    }
    cvtColor(curFrame, this->lastFrame, COLOR_BGR2GRAY);
    GaussianBlur(this->lastFrame, lastFrame, Size(21, 21), 0);
    return fl_motion;
}