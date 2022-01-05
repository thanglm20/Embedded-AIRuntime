

#include "FeatureExtractor.hpp"

FeatureExtractor::FeatureExtractor(/* args */)
{
    this->m_trafficManager = new airuntime::aiengine::its::TrafficManager();
    this->m_subtractor = new BackgroundSub();
    this->m_stickStart = cv::getTickCount() / getTickFrequency();
}

FeatureExtractor::~FeatureExtractor()
{

}
void FeatureExtractor::clearSaving()
{
    // make folder to save
    char cmd[150];
    sprintf(cmd, "rm -rf %s/*",  PATH_EXTRACTOR);
    system(cmd);
}
void FeatureExtractor::run(cv::Mat& img, int timeCounter)
{
    // count frame
    this->m_nFrame++;
    if(this->m_nFrame > MAX_FRAME_PER_DAY)
        this->m_nFrame = 0;
    // count time
    this->m_fTime = cv::getTickCount() / getTickFrequency() - this->m_stickStart;
    if(this->m_fTime > MAX_FRAME_PER_DAY)
    {
        this->m_stickStart = cv::getTickCount() / getTickFrequency();
        this->m_fTime = cv::getTickCount() / getTickFrequency() - this->m_stickStart;
    }
    
    std::cout << "Time: " << this->m_fTime << ", frame: " << this->m_nFrame << std::endl;
    std::vector<airuntime::aiengine::its::VehicleTrace> outVehicles; 
    this->m_trafficManager->run(img, outVehicles);

    this->m_subtractor->extract(img);
    
    // post process
    std::cout << "-------------------\n";
    for(auto v : outVehicles)
    {
        const int theId =  v.track_id;
        const auto p = find_if(this->m_listVehicles.begin(), this->m_listVehicles.end(), 
                                [theId] ( const Features& a ) { return (a.ID == theId);});   
        // old id                      
        if (p != this->m_listVehicles.end()) 
        {
            int i = distance(this->m_listVehicles.begin(), p);
            this->m_listVehicles[i].count++;
            ObjInfo obj;
            obj.rect = v.rect;
            obj.frame = this->m_nFrame;
            obj.time = this->m_fTime;
            this->m_listVehicles[i].info.push_back(obj);

            // saving cropped image
            char path[200];
            sprintf(path, "%s/%ld.jpg", this->m_listVehicles[i].pathSaving, this->m_nFrame);
            cv::Mat imgObj = img(v.rect).clone();
            cv::imwrite(path, imgObj);
        }
        // new id
        else
        {
            Features ft;
            ft.ID = v.track_id;
            ft.count = 1;
            ft.label = v.type;
            ft.score = v.score;
            ObjInfo obj;
            obj.rect = v.rect;
            obj.frame = this->m_nFrame;
            obj.time = this->m_fTime;
            ft.info.push_back(obj);
            sprintf(ft.pathSaving, "%s%d", PATH_EXTRACTOR, ft.ID);
            this->m_listVehicles.push_back(ft);

            // make folder to save
            char cmd[150];
            sprintf(cmd, "mkdir -p %s",  ft.pathSaving);
            system(cmd);
            // saving cropped image
            char path[200];
            sprintf(path, "%s/%ld.jpg", ft.pathSaving, this->m_nFrame);
            cv::Mat imgObj = img(v.rect).clone();
            cv::imwrite(path, imgObj);
        }
    }

    if(this->m_fTime >= timeCounter && (int)this->m_fTime % timeCounter == 0)
    {

        // get json
        for(auto v : this->m_listVehicles)
        {
            json j;
            j["label"] = v.label;
            j["id"]  = v.ID;
            j["count"] = v.count;
            for(auto o : v.info)
            {
                json b;
                b["frame"] = o.frame;
                b["time"] = o.time;
                b["box"].push_back(o.rect.x);
                b["box"].push_back(o.rect.y);
                b["box"].push_back(o.rect.width);
                b["box"].push_back(o.rect.height);
                j["frames"].push_back(b);
            }
            // outData.push_back(j); 
            char pathJson[200];
            // saving json
            sprintf(pathJson, "%s/info.json", v.pathSaving);
            std::ofstream output_file(pathJson);
            output_file << j;
            output_file.close();

        }

        //get background
        cv::Mat background = this->m_subtractor->getBackground();
        if(!background.empty())
        {
            char path[100];
            sprintf(path, "%s/background.jpg", PATH_EXTRACTOR);
            cv::imwrite(path, background);
        }


    }
    
}