/*--------------------------------------------------------------------------------------
    Module: ImageProcessor.cpp
    Author: ThangLMb
    Data: April 20, 2021
--------------------------------------------------------------------------------------*/
#include <iostream>
#include <err.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/mount.h>
#include <sys/poll.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits.h>
#include "ImageProcessor.hpp"


cv::Mat resize2SquareImage( const cv::Mat& img, int dest_width)
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( dest_width, dest_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) dest_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = dest_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( dest_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = dest_width;
        roi.width = width * scale;
        roi.x = ( dest_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}
void *xcalloc(size_t nmemb , size_t size)
{
    void *ptr = calloc(nmemb, size);
    if(!ptr)
    {
        printf("malloc error \n");
        exit(1);
    }
    return ptr;
}


unsigned char* loadYUV(const char* filename, int w, int h)
{
    unsigned char* data;
    FILE        *yuv;
    yuv = fopen(filename,"rb");
    data = (unsigned char*)malloc(w * h *1.5 * sizeof(unsigned char));
    fread(data, sizeof(unsigned char), (w * h) * 1.5, yuv);
    fclose(yuv);
    return data;
}


int save_image_yuv(const char *filename, const unsigned char *YUV, size_t y_stride, size_t uv_stride)
{
	FILE *fp = fopen(filename, "wb");
	if(!fp)
	{
		perror("Error opening yuv image for write");
		return 1;
	}
	fwrite(YUV, 1, y_stride * uv_stride * 3/2, fp);	
	fclose(fp);
	return 0;
}


unsigned char* convert_yuv2bgr( unsigned char* buf_src, int src_w, int src_h, int dest_w, int dest_h )
{
    cv::Mat nv_21_mat = cv::Mat(src_h * 1.5, src_w, CV_8UC1, buf_src);
    cv::Mat bgr_mat;
    cv::cvtColor(nv_21_mat, bgr_mat, cv::COLOR_YUV2BGR_NV21);
    cv::Mat bgr_resize;
    cv::resize(bgr_mat, bgr_resize, cv::Size(dest_w, dest_h));
    cv::Mat flat = bgr_resize.reshape(1, bgr_resize.total()*bgr_resize.channels());
    std::vector<unsigned char> vec = bgr_resize.isContinuous()? flat : flat.clone();
    //printf("Size yuv2bgr: %d\n", vec.size());
    unsigned char *bgr = &*vec.begin();
    return bgr;
}

void hough_transform(Mat& im,Mat& orig,double* skew)
{
    double max_r=sqrt(pow(.5*im.cols,2)+pow(.5*im.rows,2));
    int angleBins = 180;
    Mat acc = Mat::zeros(Size(2*max_r,angleBins),CV_32SC1);
    int cenx = im.cols/2;
    int ceny = im.rows/2;
    for(int x=1;x<im.cols-1;x++)
    {
        for(int y=1;y<im.rows-1;y++)
        {
            if(im.at<uchar>(y,x)==255)
            {
                for(int t=0;t<angleBins;t++)
                {
                    double r =(x-cenx)*cos((double)t/angleBins*CV_PI)+(y-ceny)*sin((double)t    /angleBins*CV_PI);
                    r+=max_r;
                    acc.at<int>(t,int(r))++;
                }
            }
        }
    }
    Mat thresh;
    normalize(acc,acc,255,0,NORM_MINMAX);
    convertScaleAbs(acc,acc);
    /*debug
    Mat cmap;
    applyColorMap(acc,cmap,COLORMAP_JET);
    imshow("cmap",cmap);
    imshow("acc",acc);*/

    Point maxLoc;
    minMaxLoc(acc,0,0,0,&maxLoc);
    double theta = (double)maxLoc.y/angleBins*CV_PI;
    double rho = maxLoc.x-max_r;
    if(abs(sin(theta))<0.000001)//check vertical
    {
        //when vertical, line equation becomes
        //x = rho
        double m = -cos(theta)/sin(theta);
        Point2d p1 = Point2d(rho+im.cols/2,0);
        Point2d p2 = Point2d(rho+im.cols/2,im.rows);
        //line(orig,p1,p2,Scalar(0,0,255),1);
        *skew=90;
        //cout<<"skew angle "<<" 90"<<endl;
    }else
    {
        //convert normal form back to slope intercept form
        //y = mx + b
        double m = -cos(theta)/sin(theta);
        double b = rho/sin(theta)+im.rows/2.-m*im.cols/2.;
        Point2d p1 = Point2d(0,b);
        Point2d p2 = Point2d(im.cols,im.cols*m+b);
        //line(orig,p1,p2,Scalar(0,0,255),1);
        double skewangle;
        skewangle= p1.x-p2.x>0? (atan2(p1.y-p2.y,p1.x-p2.x)*180./CV_PI):(atan2(p2.y-p1.y,p2.    x-p1.x)*180./CV_PI);
        *skew=skewangle;
        //cout<<"skew angle "<<skewangle<<endl;
    }
}

Mat preprocess1(Mat& im)
{
    Mat ret = Mat::zeros(im.size(),CV_32SC1);

    for(int x=1;x<im.cols-1;x++)
    {
        for(int y=1;y<im.rows-1;y++)
        {

            int gy = (im.at<uchar>(y-1,x+1)-im.at<uchar>(y-1,x-1))
                +2*(im.at<uchar>(y,x+1)-im.at<uchar>(y,x-1))
                +(im.at<uchar>(y+1,x+1)-im.at<uchar>(y+1,x-1));
            int gx = (im.at<uchar>(y+1,x-1) -im.at<uchar>(y-1,x-1))
                +2*(im.at<uchar>(y+1,x)-im.at<uchar>(y-1,x))
                +(im.at<uchar>(y+1,x+1)-im.at<uchar>(y-1,x+1));
            int g2 = (gy*gy + gx*gx);
            ret.at<int>(y,x)=g2;
        }
    }
    normalize(ret,ret,255,0,NORM_MINMAX);
    ret.convertTo(ret,CV_8UC1);
    threshold(ret,ret,50,255,THRESH_BINARY);
    return ret;
}

Mat preprocess2(Mat& im)
{
    // 1) assume white on black and does local thresholding
    // 2) only allow voting top is white and buttom is black(buttom text line)
    Mat thresh;
    //thresh=255-im;
    thresh=im.clone();
    adaptiveThreshold(thresh,thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, -2);
	//adaptiveThreshold(thresh,thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 9, 31);
    Mat ret = Mat::zeros(im.size(), CV_8UC1);
    for(int x=1;x<thresh.cols-1;x++)
    {
        for(int y=1;y<thresh.rows-1;y++)
        {
            bool toprowblack = thresh.at<uchar>(y-1,x)==0 ||  thresh.at<uchar>(y-1,x-1)==0     || thresh.at<uchar>(y-1,x+1)==0;
            bool belowrowblack = thresh.at<uchar>(y+1,x)==0 ||  thresh.at<uchar>(y+1,    x-1)==0 || thresh.at<uchar>(y+1,x+1)==0;

            uchar pix=thresh.at<uchar>(y,x);
            if((!toprowblack && pix==255 && belowrowblack))
            {
                ret.at<uchar>(y,x) = 255;
            }
        }
    }
    return ret;
}
Mat rototeImage(Mat& im,double thetaRad)
{
    cv::Mat rotated;
    double rskew = thetaRad* CV_PI/180;
    double nw = abs(sin(thetaRad)) * im.rows+abs(cos(thetaRad)) * im.cols;
    double nh = abs(cos(thetaRad)) * im.rows+abs(sin(thetaRad)) * im.cols;
    cv::Mat rot_mat = cv::getRotationMatrix2D(Point2d(nw * .5, nh * .5), thetaRad * 180/CV_PI, 1);
    Mat pos = Mat::zeros(Size(1,3), CV_64FC1);
    pos.at<double>(0) = (nw-im.cols) * .5;
    pos.at<double>(1) = (nh-im.rows) * .5;
    Mat res = rot_mat * pos;
    rot_mat.at<double>(0,2) += res.at<double>(0);
    rot_mat.at<double>(1,2) += res.at<double>(1);
    cv::warpAffine(im, rotated, rot_mat,Size(nw,nh), cv::INTER_LANCZOS4);
    return rotated;
}

cv::Mat filterMedianSmoot(const cv::Mat &source)
{
    cv::Mat results;
    cv::medianBlur(source, results, 3);
    return results;
}

cv::Mat filterGaussian(const cv::Mat&source)
{
    cv::Mat results;
    cv::GaussianBlur(source, results, cv::Size(3, 3), 0);
    return results;
}



cv::Mat binarize(const cv::Mat&source)
{    
    cv::Mat results;
    int blockDim = MIN( source.size().height / 4, source.size().width / 4);
    if(blockDim % 2 != 1) blockDim++;
	std::cout << blockDim;

    cv::adaptiveThreshold(source, results, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockDim, 0);
    return results;
}

cv::Mat skewImageLine (cv::Mat &img)
{
	Mat gray;
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    Mat preprocessed = preprocess2(gray);
    //imshow("preprocessed2",preprocessed);
    double skew;
    hough_transform(preprocessed, img, &skew);
    Mat rotated = rototeImage(img, skew* CV_PI/180);
	return rotated;
}
void showHistogram(Mat& img, char* namew)
{
	int bins = 256;             // number of bins
	int nc = img.channels();    // number of channels
	vector<Mat> hist(nc);       // histogram arrays
	// Initalize histogram arrays
   // printf("Hist size: %d\n", hist;
	for (int i = 0; i < hist.size(); i++)   
		hist[i] = Mat::zeros(1, bins, CV_32SC1);

	// Calculate the histogram of the image
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < nc; k++)
			{
				uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
				hist[k].at<int>(val) += 1;
			}
		}
	}
	// For each histogram arrays, obtain the maximum (peak) value
	// Needed to normalize the display later
	int hmax[3] = {0,0,0};
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < bins-1; j++)
			hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
	}

	const char* wname[3] = { "blue", "green", "red" };
	Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	vector<Mat> canvas(nc);

	// Display each histogram in a canvas
	for (int i = 0; i < nc; i++)
	{
		canvas[i] = Mat::ones(125, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins-1; j++)
		{
			line(
				canvas[i], 
				Point(j, rows), 
				Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])), 
				nc == 1 ? Scalar(200,200,200) : colors[i], 
				1, 8, 0
			);
		}

		imshow(nc == 1 ? namew : wname[i], canvas[i]);
	}
}
void imHistGray(Mat image, int histogram[])
{
 
    // initialize all intensity values to 0
    for(int i = 0; i < 256; i++)
    {
        histogram[i] = 0;
    }
 
    // calculate the no of pixels for each intensity values 
    for(int y = 0; y < image.rows; y++)
        for(int x = 0; x < image.cols; x++)
        {            
            histogram[(int)image.at<uchar>(y,x)]++;
        }
    int sizes = (*(&histogram + 1) - histogram) / sizeof(*histogram);
            
}
cv::Mat equalizeHistogram(cv::Mat &image)
{
    int histogram[256];
    imHistGray(image, histogram);
    // Calculate the size of image
    int size = image.rows * image.cols;

    // Calculate the probability of each intensity
    float PrRk[256];
    for(int i = 0; i < 256; i++)
    {
        PrRk[i] = (double)histogram[i] / size;
    }
 
    // Generate the equalized histogram
    float PsSk[256];
    for(int i = 0; i < 256; i++)
    {
        PsSk[i] = 0;
    }
 
    for(int i = 0; i < 256; i++)
        for(int j=0; j<=i; j++)
            PsSk[i] += PrRk[j];

    int final[256];
    for(int i = 0; i < 256; i++)
        final[i] = cvRound(PsSk[i]*255);

    for(int i = 0; i < 256; i++)
        for(int j=0; j<=255; j++)
            if (final[i]==final[j] && i!=j)
            {
                final[i]+=final[j];
            } 

    int final1[256];
    for(int i = 0; i < 256; i++)
    {
        final1[i]=0;
    }
    
    for(int i = 0; i < 256; i++)
    {
            final1[cvRound(PsSk[i]*255)] =cvRound(PrRk[i]*size);
    }

    for(int i = 0; i < 256; i++)
        for(int j=0; j<256; j++)
            if (final1[i]==final[j] && i!=j)
            {
                final1[i]+=final1[j];
                //cout<<"final1["<<i<<"]="<<final1[i]<<endl;
            }
    
    // Generate the equlized image
    Mat new_image = image.clone();

    for(int y = 0; y < image.rows; y++)
        for(int x = 0; x < image.cols; x++)
            new_image.at<uchar>(y,x) = saturate_cast<uchar>(final[image.at<uchar>(y,x)]);
    
    return new_image;
}
cv::Mat equalizeHistgramGray(cv::Mat &img){
    // Total number of occurance of the number of each pixels at different levels from 0 - 256
    // Flattening our 2d matrix
    int flat_img[256] = {0};
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            int index;
            index = static_cast<int>(img.at<uchar>(i,j)); // safe convertion to int
            flat_img[index]++;
        }
    }

    // calculate CDF corresponding to flat_img
    // CDF --> cumsum
    int cumsum[256]={0};
    int memory=0;
    for(int i=0; i<256; i++){
        memory += flat_img[i];
        cumsum[i] = memory;
    }

    // using general histogram equalization formula
    int normalize_img[256]={0};
    for(int i=0; i<256; i++){
    	// norm(v) = round(((cdf(v) - mincdf) / (M * N) - mincdf) * (L - 1));
        normalize_img[i] = ((cumsum[i]-cumsum[0])*255)/(img.rows*img.cols-cumsum[0]);
        normalize_img[i] = static_cast<int>(normalize_img[i]);
    }

    // convert 1d back into a 2d matrix
    cv::Mat result(img.rows, img.cols, CV_8U);
    
    Mat_<uchar>::iterator itr_result = result.begin<uchar>(); // our result
    Mat_<uchar>::iterator it_begin = img.begin<uchar>(); // beginning of the image
    Mat_<uchar>::iterator itr_end = img.end<uchar>(); // end of the image
    
    for(; it_begin!=itr_end; it_begin++){
        int intensity_value = static_cast<int>(*it_begin); // get the value and cast it into an int
        *itr_result = normalize_img[intensity_value];
        itr_result++;
    }


    return result;
}
cv::Mat equalizeHistogramRGB(const Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        vector<Mat> channels;
        split(inputImage,channels);
        Mat B,G,R;

        equalizeHist( channels[0], B );
        equalizeHist( channels[1], G );
        equalizeHist( channels[2], R );
        vector<Mat> combined;
        combined.push_back(B);
        combined.push_back(G);
        combined.push_back(R);
        Mat result;
        merge(combined,result);
        return result;
    }
    return Mat();
}
