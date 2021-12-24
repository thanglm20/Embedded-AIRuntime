/******************************************************************************** 
 Copyright (C) 2020, LE MANH THANG. All rights reserved.
 Module: SnpeInsightface.cpp
 Author: LE MANH THANG
 Created: 01/02/2021
 Description: 
********************************************************************************/
#ifndef MTCNN_H
#define MTCNN_H

struct face_landmark
{
	float x[5];
	float y[5];
};

struct face_box
{
	float x0;
	float y0;
	float x1;
	float y1;

	/* confidence score */
	float score;

	/*regression scale */

	float regress[4];

	/* padding stuff*/
	float px0;
	float py0;
	float px1;
	float py1;

	face_landmark landmark;  
};
		
static int	min_size_= 40;
static float pnet_threshold_= 0.6;
static float rnet_threshold_= 0.7;
static float onet_threshold_= 0.8;
static float factor_= 0.709;

#endif


