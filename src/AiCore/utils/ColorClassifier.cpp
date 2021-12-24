

/*
    Module: ColorClassifier.cpp
    Author: LE MANH THANG
    Created: Oct 30th, 2021
*/

#include "ColorClassifier.hpp"


ColorClassifier::ColorClassifier(/* args */)
{
}

ColorClassifier::~ColorClassifier()
{
}

ColorClassifier::m_Color ColorClassifier::GetPixelColorType(int H, int S, int V)
{

	m_Color color;
	if (V < 75)
		color = C_BLACK;
	else if (V > 190 && S < 27)
		color = C_WHITE;
	else if (S < 53 && V < 185)
		color = C_GREY;
	else
	{	// Is a color
		if (H < 5)
			color = C_RED;
		else if (H < 15)
			color = C_ORANGE;
		else if (H < 33)
			color = C_YELLOW;
		else if (H < 85)
			color = C_GREEN;
		else if (H < 102)
			color = C_AQUA;
		else if (H < 120)
			color = C_BLUE;
		else if (H < 155)
			color = C_PURPLE;
		else if (H <= 175)
			color = C_PINK;
		else	// full circle 
			color = C_RED;	// back to Red
	}
	return color;
}

ColorClassifier::m_Color ColorClassifier::GetPixelColorType(cv::Vec3b pixel)
{
	uchar H = pixel[0];  	// Hue
	uchar S = pixel[1];		// Saturation
	uchar V = pixel[2]; 	// Value (Brightness)
	return GetPixelColorType(H, S, V);
}

std::string ColorClassifier::ClassifyColor(const cv::Mat& img)
{
		
		cv::Mat imageShirtHSV;
		cv::cvtColor(img, imageShirtHSV, cv::COLOR_RGB2HSV);
			// (note that OpenCV stores RGB images in B,G,R order.
		//ASSERT(imageShirtHSV.data, "ERROR: Couldn't convert Shirt image from BGR2HSV.")

			//std::cout << "Determining color type of the shirt" << endl;
			int h = imageShirtHSV.rows;				// Pixel height
		int w = imageShirtHSV.cols;				// Pixel width
		int rowSize = imageShirtHSV.step;		// Size of row in bytes, including extra padding

		// Create an empty tally of pixel counts for each color type
		int tallyColors[ColorClassifier::m_Color::NUM_COLOR_TYPES];
		for (int i = 0; i < ColorClassifier::m_Color::NUM_COLOR_TYPES; i++)
		{
			tallyColors[i] = 0;
		}

		// Scan the shirt image to find the tally of pixel colors
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x<w; x++)
			{
				cv::Vec3b pixel = imageShirtHSV.at<cv::Vec3b>(y, x);
				if ((pixel[0] != 0) && (pixel[1] != 0) && (pixel[2] != 0))
				{
				// Determine what type of color the HSV pixel is.
				int ctype = GetPixelColorType(pixel);

				//std::cout << TGMTcolor::GetH(pixel) << endl;
				// Keep count of these colors.
				//cout << ctype <<endl;
				tallyColors[ctype]++;
				}
			}
		}
		
		// Print a report about color types, and find the max tally
		int tallyMaxIndex = 0;
		int tallyMaxCount = -1;
		int pixels = w * h;
		for (int i = 0; i < ColorClassifier::m_Color::NUM_COLOR_TYPES; i++)
		{
			int v = tallyColors[i];

			if (v > tallyMaxCount)
			{
				tallyMaxCount = tallyColors[i];
				tallyMaxIndex = i;
			}
		}

		int percentage = (tallyMaxCount * 100 / pixels);
		const char* color = "";
		color = ColorNames[tallyMaxIndex].c_str();
		// printf("Detected color: %s with %d percent", color, percentage);
		return color;
}
