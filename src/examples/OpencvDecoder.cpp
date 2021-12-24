

#include "OpencvDecoder.hpp"


OpencvDecoder::OpencvDecoder()
{

}
OpencvDecoder::~OpencvDecoder(){

}
int OpencvDecoder::open( const char* file_video){
    this->m_cap = cv::VideoCapture(file_video);
    if(!this->m_cap.isOpened())
    {
        std::cout << "Opening video error\n";
        return -1;
    }
    return 0;
}
cv::Mat OpencvDecoder::getFrame()
{
    cv::Mat frame;
    this->m_cap >> frame;
    return frame;
}