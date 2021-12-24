/*
    Module: NcnnExecutor.cpp
    Author: LE MANH THANG
    Created: Dec 21th, 2021
*/
#include "NcnnExecutor.hpp"

static int num_line = 0;
static std::string read_file(const char* path, bool count_line)
{
    std::string result = "";
    std::ifstream f(path);
    //check file exist | true = exist
    if (!f.good()) return result;
    else
    {
        std::string temp;
        while (std::getline(f, result))
        {
            temp.append(result);
            // std::cout << result;
            if (count_line == true)
            {
                num_line++;
            }
        }
        result = temp;
        // Close the file
        f.close();
    }
    num_line = 0;
    return result;
}

NcnnExecutor::NcnnExecutor(airuntime::DeviceType device, 
                            airuntime::AlgTypeAI algType,
                            std::string pathLabel,
                            std::string modelWeight, 
                            std::string modelParam
                            ) 
    : AIExecutor(airuntime::ExecutorType::NCNN, device, algType, pathLabel, modelWeight, modelParam)
    {
        cout << "==============> NCNN Executor <=================\n";
        cout << "Model weight: " << getModelWeight() << endl;
        cout << "Model param: " << getModelParam() << endl;
        cout << "Label: " << getPathLabel() << endl;
        cout << "================================================\n";
    }

NcnnExecutor::~NcnnExecutor()
{
    cout << "Called NcnnExecutor destructor\n";
}

STATUS NcnnExecutor::init(){
    cout << "Initiated NCNN Executor successfully\n";
    return STATUS::SUCCESS;
}

STATUS NcnnExecutor::run(const Mat& img, 
                            vector<ObjectTrace>& objects,
                            float threshold)
{

    return STATUS::SUCCESS;
}

STATUS NcnnExecutor::release(){

    cout << "Released NCNN Executor successfully\n";
    return STATUS::SUCCESS;
}


