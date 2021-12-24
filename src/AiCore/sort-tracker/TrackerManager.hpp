/*
    Module: Object Tracking
    Author: Le Manh Thang
    Created: August 26th, 2021
*/
#ifndef TrackerManager_hpp
#define TrackerManager_hpp
#include "Hungarian.h"
#include "KalmanTracker.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../AITypeData.hpp"
typedef cv::Point_<float> Point_t;

struct TrajectoryPoint
{
    TrajectoryPoint()
    {
    }
    TrajectoryPoint(const Point_t& prediction) : m_prediction(prediction)
    {
    }
    Point_t m_prediction;
};

class Trace
{
public:

    const Point_t& operator[](size_t i) const
    {
        return m_trace[i].m_prediction;
    }

    Point_t& operator[](size_t i)
    {
        return m_trace[i].m_prediction;
    }

    const TrajectoryPoint& at(size_t i) const
    {
        return m_trace[i];
    }
    
    size_t size() const
    {
        return m_trace.size();
    }

    void push_back(const Point_t& prediction)
    {
        m_trace.emplace_back(prediction);
    }

    void pop_front(size_t count)
    {
        if (count < size())
        {
            m_trace.erase(m_trace.begin(), m_trace.begin() + count);
        }
        else
        {
            m_trace.clear();
        }
    }

private:
    std::deque<TrajectoryPoint> m_trace;
};

struct TrackingTrace
{
    Trace m_trace;                     // Trajectory
    Rect2f m_rect;                     // box of object track
    string m_type;                     // Objects type name or empty value
    size_t m_ID = 0;                   // Objects ID
    bool isOutOfFrame = false;             // if object Out of Frame
};


class TrackerManager
{
private:
    /* data */
    int max_trace_length = 50;
    int max_age = 10;
    int min_confirmed = 2;
public:
    TrackerManager(Rect2f rect, string typeObject);
    ~TrackerManager();
    KalmanTracker tracker;
    TrackingTrace tracer;
    TrackingTrace get() const;
    Rect2f predict();
    void update(Rect2f rect);
    bool isConfirmed();
    bool isDeleted();
    bool isTentative();
    bool isOutOfFrame();
};

#endif
