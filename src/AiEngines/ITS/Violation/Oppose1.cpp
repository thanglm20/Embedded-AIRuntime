

/*
    Module: Oppose
    Author: Le Manh Thang
    Created: Oct 04, 2021
*/

#include "Oppose.hpp"

inline bool checkDirection(LinearEquationStartLine factorStartLine, Point pointCheck, Point pointRailEnd)
{

    if((factorStartLine.A * pointRailEnd.x + factorStartLine.B * pointRailEnd.y + factorStartLine.C) * 
        (factorStartLine.A * pointCheck.x  + factorStartLine.B * pointCheck.y + factorStartLine.C) > 0)
        return true;
           
    else
        return false;   
}

inline LinearEquationStartLine findEquationLine(vector<Point> startLine)
{
    LinearEquationStartLine equation;
    int x1 = startLine.at(0).x;
    int y1 = startLine.at(0).y;
    int x2 = startLine.at(1).x;
    int y2 = startLine.at(1).y;

    // d parallel with Ox => y = a
    if(y2 == y1)
    {
        equation.A = 0;
        equation.B = -1;
        equation.C = y2;

    }
    // d parallel with Oy => x = a
    if(x2 == x1)
    {
        equation.A = -1;
        equation.B = 0;
        equation.C = x1;
    }
    if(y2 != y1 && x2 != x1)
    {
        equation.A = (y2 - y1) * 1.0 / (x2 - x1);
        equation.B = -1;
        equation.C = y1  - equation.A * x1;
    }
    
    return equation;
}

// find point inline rail of start line and rail of end line
// distance point to start line equal d
// point is the same direction with rail of end line 
inline Point2d findPoint( LinearEquationStartLine equationStartLine, float d, Point tail, Point head)
{
    // find linear equation between rail of start and end line
    vector<Point> vecLine{tail, head};
    LinearEquationStartLine equationTail = findEquationLine(vecLine);
     
    //      |ax + by + c| = d * sqrt(a^2 + b^2)
    //       Ax + By + C = 0
    // find Point
    auto findVar = [](double a1, double b1, double c1,
    double& a2, double b2, double c2) -> Point2d
    {
        double x = (b1*c2 - b2 * c1) / (a1*b2 - a2 * b1);
        double y = (c1*a2 - c2 * a1) / (a1*b2 - a2 * b1);
        return Point2d(x, y);
    };

    Point2d point(0, 0);
    Point2d point1 = findVar(equationStartLine.A, equationStartLine.B, 
    equationStartLine.C - d * sqrt(pow(equationStartLine.A, 2) + pow(equationStartLine.B, 2))
    , equationTail.A, equationTail.B, equationTail.C);
    if(checkDirection(equationStartLine, point1, head))
        return point1;
        
    Point2d point2 = findVar(equationStartLine.A, equationStartLine.B, 
    equationStartLine.C + d * sqrt(pow(equationStartLine.A, 2) + pow(equationStartLine.B, 2))
    , equationTail.A, equationTail.B, equationTail.C);
    if(checkDirection(equationStartLine, point2, head))
        return  point2;

    return point;
}   

inline vector<VecLine> calMeshLine(vector<Point> startLine, vector<Point> endLine)
{
    vector<VecLine> listLine;
    // find line to mesh
    LinearEquationStartLine equationStartLine = findEquationLine(startLine);


    double dTailToRail = fabs(equationStartLine.A * endLine.at(0).x + equationStartLine.B * endLine.at(0).y + equationStartLine.C)
                / sqrt((pow(equationStartLine.A, 2) + pow(equationStartLine.B, 2)));
    
    double dHeadToHead = fabs(equationStartLine.A * endLine.at(1).x + equationStartLine.B * endLine.at(1).y + equationStartLine.C)
                / sqrt((pow(equationStartLine.A, 2) + pow(equationStartLine.B, 2)));

    double dTail = 0;
    double dHead = 0;
    listLine.push_back(VecLine{startLine.at(0), startLine.at(1)});
    for(int i = NUMBER_LINE_CHECK; i > 1; i--)
    {
        dTail += dTailToRail / NUMBER_LINE_CHECK;     
        dHead += dHeadToHead / NUMBER_LINE_CHECK;   
        VecLine lineCheck;
        lineCheck.tail = findPoint(equationStartLine, dTail, startLine.at(0), endLine.at(0));
        lineCheck.head = findPoint(equationStartLine, dHead, startLine.at(1), endLine.at(1));      
        listLine.push_back(lineCheck);
    }
    listLine.push_back(VecLine{endLine.at(0), endLine.at(1)});
    return listLine;
}


Oppose::Oppose()
{
    
    this->m_detector = new VehicleDetector();
    this->tracker = new ObjectTracking();
}

Oppose::Oppose(/* args */settingsOppose settings)
{
    
    this->m_detector = new VehicleDetector();
    this->tracker = new ObjectTracking();
}

Oppose::~Oppose()
{
    if(this->m_detector) delete this->m_detector;
    if(this->tracker) delete this->tracker;
}


int Oppose::init(settingsOppose settings)
{
    this->settings = settings;
    
    // cal linear equation of start line
    this->factorStartLine = findEquationLine(this->settings.startLine);
    
    // check start line and end line are valid
    if(( this->factorStartLine.A * this->settings.endLine.at(0).x + this->factorStartLine.B *
        this->settings.endLine.at(0).y + this->factorStartLine.C) * 
        (this->factorStartLine.A * this->settings.endLine.at(1).x + this->factorStartLine.B * 
        this->settings.endLine.at(1).y + this->factorStartLine.C) < 0)
        {
            LOG_FAIL("Start line and end line are not valid");
            return STATUS::INVALID_ARGS;
        }
    // get mesh line
    this->settings.listLine = calMeshLine(this->settings.startLine, this->settings.endLine);

    return STATUS::SUCCESS;
}
int Oppose::set(settingsOppose settings)
{
    this->settings = settings;
    // cal linear equation of start line
    this->factorStartLine = findEquationLine(this->settings.startLine);
    
    // check start line and end line are valid
    if(( this->factorStartLine.A * this->settings.endLine.at(0).x + this->factorStartLine.B * 
        this->settings.endLine.at(0).y + this->factorStartLine.C) * 
        (this->factorStartLine.A * this->settings.endLine.at(1).x + this->factorStartLine.B * 
        this->settings.endLine.at(1).y + this->factorStartLine.C) < 0)
        {
            LOG_FAIL("Start line and end line are not valid");
            return STATUS::FAIL;
        }
    // get mesh line
    this->settings.listLine = calMeshLine(this->settings.startLine, this->settings.endLine);
    return STATUS::SUCCESS;
}
int Oppose::update(Mat& frame, vector<outDataOppose>& outData)
{
    try
    {
        Mat img;
        frame.copyTo(img);
        int widthFrame = img.cols;
        int heightFrame = img.rows;
        outData.clear();
     
        // detect
        auto start = std::chrono::high_resolution_clock::now();    
        std::vector<ObjectTrace> detected;
		if(this->m_detector->run(img, detected, THRES_DETECT_VEHICLE) != STATUS::SUCCESS)
        {
            LOG_FAIL("Execute Anpr detector failed");
            return STATUS::FAIL;
        }
        auto end = std::chrono::high_resolution_clock::now();    
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // cout << "Time detect: " <<  duration.count() << endl;
        // tracking
        vector<TrackingTrace> tracks;
        this->tracker->process(detected, tracks);
        
        //delete object which is abandoned
        for(auto it = this->listTracked.begin(); it != this->listTracked.end();)
        {
            const int theId =  (*it).track_id;
            const auto p = find_if(tracks.begin(), tracks.end(), 
                                        [theId] ( const TrackingTrace& a ) { return (a.m_ID == theId);}); 
            if (p == tracks.end() && it != this->listTracked.end())
                it = this->listTracked.erase(it);                
            else 
                it++;
        }
        // check opposition
        for (auto &track: tracks) 
        {
            
            if(!track.isOutOfFrame)
            {

                // draw box
                // rectangle(img, track.m_rect, Scalar(255, 255, 255), 1, 8);
                // char text[100];
			    // sprintf(text,"%d:%s", (int)track.m_ID, track.m_type.c_str());
			    // cv::putText(img, text, cv::Point(track.m_rect.x, track.m_rect.y), FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
                
                // if object is not in list of allowed, then abort
                if(find(this->settings.allowedObjects.begin(), this->settings.allowedObjects.end(), track.m_type) 
                != this->settings.allowedObjects.end())
                    continue;

                // find object in tracked list                 
                int theId = track.m_ID;
                const auto p = find_if(this->listTracked.begin(), this->listTracked.end(), [theId] (const outDataOppose& a) {return theId == a.track_id;});
                // if found
                if(p != this->listTracked.end())
                {
                    int i = distance(this->listTracked.begin(), p);
                    this->listTracked[i].rect = track.m_rect;
                    // check start 
                    for(int c =  this->listTracked[i].indexFirstLine + 1; c < this->settings.listLine.size(); c++)
                    {
                        VecLine lineCheck = this->settings.listLine[c];
                        Point2f tail((float) lineCheck.tail.x/widthFrame, (float) lineCheck.tail.y/heightFrame);
                        Point2f head((float) lineCheck.head.x/widthFrame, (float) lineCheck.head.y/heightFrame); 
                        int direction = this->tracker->checkCrossline(track, widthFrame, heightFrame, RoadLine(tail, head, 0));
                        if(direction == this->listTracked[i].direction)
                        {
                            this->listTracked[i].isNewEvent = true;
                            this->listTracked[i].isTentative = false;
                            //rectangle(img, track.m_rect, Scalar(255, 0, 0), 5, 8);
                        }
                    }                 
                }
                //if not found => new object
                else
                {
                    // check start 
                    for(int c = 0; c < this->settings.listLine.size(); c++)
                    {
                        VecLine lineCheck = this->settings.listLine[c];
                        Point2f tail((float) lineCheck.tail.x/widthFrame, (float) lineCheck.tail.y/heightFrame);
                        Point2f head((float) lineCheck.head.x/widthFrame, (float) lineCheck.head.y/heightFrame); 
                        int direction = this->tracker->checkCrossline(track, widthFrame, heightFrame, RoadLine(tail, head, 0));        
                        LinearEquationStartLine lineDirection = findEquationLine(vector<Point>{lineCheck.tail,lineCheck.head});      
                        if(direction  && checkDirection(lineDirection, track.m_trace[track.m_trace.size() - 1], this->settings.endLine.at(0))) 
                        {
                            // cout << "DIR: " << direction << ", ID: " << track.m_ID << ", Position: " << track.m_trace[track.m_trace.size() - 1] << endl;
                            line(img, lineCheck.tail, lineCheck.head, Scalar( 0, 255, 255), 3, LINE_AA);
                            //rectangle(img, track.m_rect, Scalar(0, 255, 255), 5, 8);
                            outDataOppose out;
                            out.indexFirstLine = c;
                            out.direction = direction;
                            out.track_id = track.m_ID;
                            out.rect = track.m_rect;
                            out.label = track.m_type;
                            out.isTentative = true;                        
                            this->listTracked.push_back(out);
                        }
                    }                    
                }
            }
            else
            {
                const int theId =  track.m_ID;
                const auto p = find_if(this->listTracked.begin(), this->listTracked.end(), 
                                        [theId] ( const outDataOppose& a ) { return (a.track_id == theId);});                         
                if (p != this->listTracked.end()) 
                {
                    int dist = distance(this->listTracked.begin(), p);
                    this->listTracked[dist].isOutOfFrame = true;
                    
                }
            } 
        }
        // abort object out of frame 
        for(auto out : this->listTracked)
            if(!out.isOutOfFrame)
                outData.push_back(out);

        //draw line
        // for(int i = 0; i < this->settings.listLine.size(); i++)
        //     line(img, this->settings.listLine[i].tail, this->settings.listLine[i].head, Scalar( 0, 0, 255), 1, LINE_AA);     

        // cv::putText(img, "START", this->settings.startLine.at(0), FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
        // line(img, this->settings.startLine.at(0), this->settings.startLine.at(1) , Scalar( 0, 0, 255), 2, LINE_AA);
        // cv::putText(img, "END", this->settings.endLine.at(0), FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 0, 255), 1);
        // line(img, this->settings.endLine.at(0), this->settings.endLine.at(1) , Scalar( 0, 0, 255), 2, LINE_AA);

        frame = img;
        //imshow("Oppose", img); 
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return STATUS::SUCCESS;
}
