
#ifndef ImageProcessor_h
#define ImageProcessor_h
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


unsigned char* loadYUV(const char* filename, int w, int h);
cv::Mat resize2SquareImage( const cv::Mat& img, int dest_width);
cv::Mat skewImageLine (cv::Mat &img);
cv::Mat equalizeHistogramRGB(const Mat& inputImage);
cv::Mat equalizeHistgramGray(cv::Mat &img);

#endif

