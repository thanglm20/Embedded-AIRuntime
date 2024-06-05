# Introduction
This project implement and deploy many embedded AI framework for edge devices, such as: NCNN, Paddle, MNN, SNPE, and Tensorflow lite, ...
Also, develop some tasks: object detection, auto plate recognition, OCR, object tracking.
# Requirements
1. paddle-lite 2.9
	 
2. adk r20b

3. Opencv

4. snpe 1.52

std c++14
OpenCV 4.5.0 built from source with c++.
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev x264 v4l-utils libhdf5-dev libprotobuf-dev protobuf-compiler git libgtk2.0-dev libgtk-3-dev

# Build & Run
Some commands are:

mkdir build
cd build
cmake ..
make
./tracker
# References
ncnn: https://github.com/Tencent/ncnn
snpe: 
paddle: 

