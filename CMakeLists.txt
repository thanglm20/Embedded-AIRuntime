
# Module: CMakelists.txt
# Author: LE MANH THANG
# Created: 10/01/2021
# Description: 
cmake_minimum_required(VERSION 3.10.2)

enable_language(C)
enable_language(CXX)

project (ai_infer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

#  # #add lib opencv
# set( OPENCV_LIB libopencv_world.so)
# link_directories(${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/lib)
# include_directories( ${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/include/opencv4)

# # #add lib opencv
# set( OPENCV_HASH_LIB libopencv_img_hash.so)
# link_directories(${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/lib)
# include_directories(${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/include/opencv4)

# #add lib snpe
set( SNPE_LIB libSNPE.so)
link_directories(${CMAKE_SOURCE_DIR}/libSNPE-prebuilt/lib/x86_64-linux-clang)
include_directories(${CMAKE_SOURCE_DIR}/libSNPE-prebuilt/include/zdl)

set( libomp libomp.so)
link_directories(${CMAKE_SOURCE_DIR}/libSNPE-prebuilt/lib/x86_64-linux-clang)
set( libHtpPrepare libHtpPrepare.so)
link_directories(${CMAKE_SOURCE_DIR}/libSNPE-prebuilt/lib/x86_64-linux-clang)

#add lib ncnn
set( NCNN_LIB libncnn.a)
link_directories(${CMAKE_SOURCE_DIR}/libncnn-linux_x86_64/lib)
include_directories(${CMAKE_SOURCE_DIR}/libncnn-linux_x86_64/include)


#add lib paddle-lite
set( PADDLE_LIB libpaddle_inference.so)
link_directories(/home/thanglmb/Documents/PaddleOCR/deploy/libs/paddle_inference/paddle/lib)
include_directories(/home/thanglmb/Documents/PaddleOCR/deploy/libs/paddle_inference/paddle/include)

include_directories(${CMAKE_SOURCE_DIR}/src)


file( GLOB SOURCES  
src/examples/main.cpp
src/examples/OpencvDecoder.cpp
src/examples/DecoderThread.cpp
src/examples/FrameManager.cpp
src/examples/Traffic.cpp
src/examples/Anpr.cpp
src/examples/Extractor.cpp

src/AiCore/AIUserFactory.cpp
src/AiCore/snpe/SnpeExecutor.cpp
src/AiCore/snpe/SnpeCommLib.cpp
src/AiCore/snpe/Util.cpp
src/AiCore/snpe/SnpeMobilenetSSD.cpp
src/AiCore/ncnn/NcnnDetector.cpp
src/AiCore/ncnn/NcnnExecutor.cpp
src/AiCore/sort-tracker/Hungarian.cpp
src/AiCore/sort-tracker/KalmanTracker.cpp
src/AiCore/sort-tracker/ObjectTracker.cpp
src/AiCore/sort-tracker/TrackerManager.cpp
src/AiCore/sort-tracker/RoadLine.cpp
src/AiCore/sort-tracker/ObjectTracking.cpp

src/AiEngines/ITS/Anpr/PaddleOCR-Linux-x64/*.cpp
src/AiEngines/ITS/Anpr/AnprDetector.cpp
src/AiEngines/ITS/Anpr/AnprRecognizer.cpp

src/AiEngines/FeatureExtractor/BackgroundSubtractor.cpp
src/AiEngines/FeatureExtractor/FeatureExtractor.cpp

src/AiEngines/ITS/TrafficManager.cpp
src/AiEngines/ITS/Violation/Oppose.cpp
)
                    
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}

${OpenCV_LIBS}
${SNPE_LIB}
${NCNN_LIB}
${PADDLE_LIB}
${libHtpPrepare}
${libomp}
 -Wall -lm -lpthread -lstdc++ -std=c++14 -lgomp -Wcpp -fPIC -march=x86-64
) 
