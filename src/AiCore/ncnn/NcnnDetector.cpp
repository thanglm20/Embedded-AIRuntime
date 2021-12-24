/******************************************************************************** 
 Module:    NcnneRuntime.Cpp
 Author:    LE MANH THANG
 Created:   06/01/2021
 Modify:    HieuPV - 02/02/2021
 Description: 
********************************************************************************/
#include "NcnnDetector.hpp"


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
NcnnDetector::NcnnDetector()
{
    this->ncnnNet = new ncnn::Net();
    #if(USE_VULKAN_COMPUTE == 1)
    this->g_blob_pool_allocator_detect = new ncnn::UnlockedPoolAllocator();
    this->g_workspace_pool_allocator_detect = new ncnn::PoolAllocator();
    this->g_blob_pool_allocator_detect->set_size_compare_ratio(0.0f);
    this->g_workspace_pool_allocator_detect->set_size_compare_ratio(0.5f);
    #endif
}
NcnnDetector::~NcnnDetector()
{
    if (this->ncnnNet) 
        delete this->ncnnNet;
    #if(USE_VULKAN_COMPUTE == 1)
    if (this->g_blob_pool_allocator_detect)
        delete this->g_blob_pool_allocator_detect;
    if (this->g_workspace_pool_allocator_detect)
        delete this->g_workspace_pool_allocator_detect;
    //HieuPV add code
    // them lenh giai phong
    if (this->g_vkdev != nullptr)
        delete this->g_vkdev;
    if (this->g_blob_vkallocator)
        delete this->g_blob_vkallocator;
    if (this->g_staging_vkallocator)
        delete this->g_staging_vkallocator;
    #endif
}
int NcnnDetector::initNcnnNetwork(const char* model_bin, const char* model_param)
{
    this->ncnnNet->opt.lightmode = true;
    this->ncnnNet->opt.num_threads = 4; //You need to compile with libgomp for multi thread support
    int ret = this->ncnnNet->load_param(model_param);
    if (ret != 0)
    {
        printf("\nLoading model param error\n");
        return ret;        
    }
    ret = this->ncnnNet->load_model(model_bin);
    if (ret != 0)
    {
        printf("\nLoading model bin error\n");
        return ret;
    }
    std::string temp = read_file(model_param, false);
    if (!temp.empty())
    {
        std::regex re("data 0=([0-9]+) 1=([0-9]+) 2=[0-9]{1}");
        std::smatch m;
        std::regex_search(temp, m, re);
        if (!m.empty())
        {
            for (auto i_m : m)
            {
                if (std::regex_match(i_m.str(), re))
                {
                    try
                    {
                        std::string s_w = i_m.str().substr(7, 9);
                        std::string s_h = i_m.str().substr(13, 15);
                        this->width_model = std::stoi(s_w);
                        this->height_model = std::stoi(s_h);
                    }
                    catch(const std::exception& e)
                    {
                        this->width_model = 416;
                        this->height_model = 416;
                    }
                    
                }
            }
        }
    }
    printf( "Ncnn network input: %d, %d \n", this->width_model, this->height_model);
    return 0;
}
int NcnnDetector::initNcnnNetwork(const char* model_bin, const char* model_param, std::string target_device )
{


    this->ncnnNet->opt.lightmode = true;
    this->ncnnNet->opt.num_threads = 4; //You need to compile with libgomp for multi thread support
    #if(USE_VULKAN_COMPUTE == 1)
    this->ncnnNet->opt.blob_allocator = this->g_blob_pool_allocator_detect;
    this->ncnnNet->opt.workspace_allocator = this->g_workspace_pool_allocator_detect;
    if(target_device == "GPU")
    {

        this->g_vkdev = ncnn::get_gpu_device(0);
        this->g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        this->g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);

        this->ncnnNet->opt.blob_vkallocator = this->g_blob_vkallocator;
        this->ncnnNet->opt.workspace_vkallocator = this->g_blob_vkallocator;
        this->ncnnNet->opt.staging_vkallocator = this->g_staging_vkallocator;

        this->ncnnNet->opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support
        this->ncnnNet->opt.use_winograd_convolution = true;
        this->ncnnNet->opt.use_sgemm_convolution = true;
        this->ncnnNet->opt.use_fp16_packed = true;
        this->ncnnNet->opt.use_fp16_storage = true;
        this->ncnnNet->opt.use_fp16_arithmetic = true;
        this->ncnnNet->opt.use_packing_layout = true;
        this->ncnnNet->opt.use_shader_pack8 = false;
        this->ncnnNet->opt.use_image_storage = false;
    }
    #endif
    int ret = this->ncnnNet->load_param(model_param);
    if (ret != 0)
    {
        printf("\nLoading model param error\n");
        return ret;        
    }
    ret = this->ncnnNet->load_model(model_bin);
    if (ret != 0)
    {
        printf("\nLoading model bin error\n");
        return ret;
    }
    std::string temp = read_file(model_param, false);
    if (!temp.empty())
    {
        std::regex re("data 0=([0-9]+) 1=([0-9]+) 2=[0-9]{1}");
        std::smatch m;
        std::regex_search(temp, m, re);
        if (!m.empty())
        {
            for (auto i_m : m)
            {
                if (std::regex_match(i_m.str(), re))
                {
                    try
                    {
                        std::string s_w = i_m.str().substr(7, 9);
                        std::string s_h = i_m.str().substr(13, 15);
                        this->width_model = std::stoi(s_w);
                        this->height_model = std::stoi(s_h);
                    }
                    catch(const std::exception& e)
                    {
                        this->width_model = 416;
                        this->height_model = 416;
                    }                    
                }
            }
        }
    }
    printf( "Ncnn network input: %d, %d \n", this->width_model, this->height_model);
    return 0;
}
int NcnnDetector::executeNcnnDetector(const cv::Mat& img, std::vector<std::string>& labels, std::vector<ObjectTrace>& objects, float thres_detect)
{
    #if(USE_VULKAN_COMPUTE == 1)
    this->g_blob_pool_allocator_detect->clear();
    this->g_workspace_pool_allocator_detect->clear();
    if (this->ncnnNet->opt.use_vulkan_compute)
    {
        this->g_blob_vkallocator->clear();
        this->g_staging_vkallocator->clear();
        this->ncnnNet->set_vulkan_device(this->g_vkdev);
    }
    #endif
    int img_w = img.cols;
    int img_h = img.rows;
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, this->width_model, this->height_model);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = this->ncnnNet->create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);
        /*HieuPV fix crash*/
        try
        {
            if(values[1] >= thres_detect )
            {
                ObjectTrace object;
                object.label        = labels[(int)values[0] - 1];
                object.obj_id       = (int)values[0] - 1;
                object.score        =  values[1] ;
                object.rect.x       = ((values[2] * img_w) < 0) ? 0 : (values[2] * img_w);
                object.rect.y       = ((values[3] * img_h) < 0) ? 0 : (values[3] * img_h); 
                object.rect.width   = values[4] * img_w - object.rect.x;
                if(object.rect.width + object.rect.x > img_w)
                    object.rect.width = img_w - object.rect.x;
                object.rect.height  = values[5] * img_h - object.rect.y;
                if(object.rect.height + object.rect.y > img_h)
                    object.rect.height = img_h - object.rect.y;
                objects.push_back(object);
            }
        }
        catch(const std::exception& e)
        {
            objects.clear();
            return -1;
        }
    }

    return 0;
}
