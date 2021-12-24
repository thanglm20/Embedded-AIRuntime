/******************************************************************************** 
Copyright (C) 2008-2009, SmartHome Co., Ltd. All rights reserved.
Product: AICAM
Module: 
Version: 1.0
Author: 
Created: 
Modified: 
    <Name>
    <Date>
    <Change>
Released: 
Description: 
Note: <Note>
********************************************************************************/
/*-----------------------------------------------------------------------------*/
/* Header inclusions                                                           */
/*-----------------------------------------------------------------------------*/
#ifndef MXNETEXTRACT_hpp
#define MXNETEXTRACT_hpp

#include <opencv2/opencv.hpp>
#include "mxnet/c_predict_api.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "../AiCore/face_recognizer/comm_lib.hpp"

struct class_info
{
	double min_distance;
	int index;
};

class Mxnet_extract 
{
public:
	Mxnet_extract(){
		;
	}
	~Mxnet_extract()
	{
		if(pred_feature)
		    MXPredFree(pred_feature);
	}
	int LoadModel(const std::string & fname, std::vector<char>& buf)
	{
		std::ifstream fs(fname, std::ios::binary | std::ios::in);

		if (!fs.good())
		{
			std::cerr << fname << " does not exist" << std::endl;
			return -1;
		}

		fs.seekg(0, std::ios::end);
		int fsize = fs.tellg();

		fs.seekg(0, std::ios::beg);
		buf.resize(fsize);
		fs.read(buf.data(), fsize);

		fs.close();

		return 0;

	}

	int LoadExtractModule(const std::string& param_file, const std::string& json_file,
		int batch, int channel, int input_h, int input_w)
	{

		std::vector<char> param_buffer;
		std::vector<char> json_buffer;

		if (LoadModel(param_file, param_buffer)<0)
			return -1;

		if (LoadModel(json_file, json_buffer)<0)
			return -1;

		int device_type = 1;
		int dev_id = 0;
		mx_uint  num_input_nodes = 1;
		const char * input_keys[1];
		const mx_uint input_shape_indptr[] = { 0, 4 };
		const mx_uint input_shape_data[] = {
			static_cast<mx_uint>(batch),
			static_cast<mx_uint>(channel),
			static_cast<mx_uint>(input_h),
			static_cast<mx_uint>(input_w)
		};

		input_keys[0] = "data";

		int ret = MXPredCreate(json_buffer.data(),
			param_buffer.data(),
			param_buffer.size(),
			device_type,
			dev_id,
			num_input_nodes,
			input_keys,
			input_shape_indptr,
			input_shape_data,
			&pred_feature
		);
		
		return ret;
	}


	cv::Mat extractFeature(const cv::Mat& img)
	{

		int width = img.cols;
		int height = img.rows;

        cv::Mat img_rgb(height, width, CV_32FC3);
		img.convertTo(img_rgb, CV_32FC3);
		cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);

		std::vector<float> input(3 * height * width);
		std::vector<cv::Mat> input_channels;
		
		set_input_buffer(input_channels, input.data(), height, width);
		cv::split(img_rgb, input_channels);

		MXPredSetInput(pred_feature, "data", input.data(), input.size());
		MXPredForward(pred_feature);

		mx_uint *shape = NULL;
		mx_uint shape_len = 0;

		MXPredGetOutputShape(pred_feature, 0, &shape, &shape_len);

		int feature_size = 1;
		for (unsigned int i = 0;i<shape_len;i++)
			feature_size *= shape[i];
		std::vector<float> feature(feature_size);

		MXPredGetOutput(pred_feature, 0, feature.data(), feature_size);
		
		cv::Mat output = cv::Mat(feature, true).reshape(1, 1);
		cv::normalize(output, output);
	
		return output;
	}
	class_info classify(const cv::Mat& img, const  cv::Mat& cmp)
	{
		int rows = cmp.rows;
		cv::Mat broad;
		cv::repeat(img, rows, 1, broad);
		broad = broad - cmp;
		cv::pow(broad,2,broad);
		cv::reduce(broad, broad, 1, cv::REDUCE_SUM);

		double dis;
		cv::Point point;
		cv::minMaxLoc(broad, &dis, 0, &point, 0);

		return class_info{dis, point.y};
	}

private:
	PredictorHandle pred_feature;
};

#endif
