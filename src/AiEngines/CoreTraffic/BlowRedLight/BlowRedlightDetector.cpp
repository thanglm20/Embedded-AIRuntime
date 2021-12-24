#include "BlowRedlightDetector.hpp"

BlowRedLightDetector::BlowRedLightDetector() {
    this->tracking = new ObjectTracking();
    this->objNames = loadObjectNames("../model/traffic/traffic.names");
    this->listVehicle.clear();
}

BlowRedLightDetector::~BlowRedLightDetector() {

}

void BlowRedLightDetector::setData(vector<vector<Point>> regions, vector<string> label, int allowRight, int allowLeft) {
    this->leftLight.clear();
    this->standardLight.clear();
    this->isLeftLine = false;
    this->isRightLine = false;
    if ((regions.size() != 0) && (label.size() == regions.size())) {
        for (int index = 0; index < regions.size(); index++) {
            if (label[index] == "0") {
                this->leftLight.push_back(regions[index]);
            }          
            else if (label[index] == "1") {
                this->standardLight.push_back(regions[index]);
            }
            else if (label[index] == "2") {
                this->beginLine = regions[index];
            }
            else if (label[index] == "3") {
                this->endLine = regions[index];
            }
            else if (label[index] == "4") {
                this->isLeftLine = true;
                this->leftLine = regions[index];
            }
            else if (label[index] == "5") {
                this->isRightLine = true;
                this->rightLine = regions[index];
            }
            else if (label[index] == "6") {
                this->fullLane = regions[index];
            }
            else if (label[index] == "7") {
                this->straightLane = regions[index];
            }
            else if (label[index] == "8") {
                this->straight_leftLane = regions[index];
            }
            else if (label[index] == "9") {
                this->straight_rightlLane = regions[index];
            }
            else if (label[index] == "11") {
                this->leftLane = regions[index];
            }
            else if (label[index] == "10") {
                this->rightLane = regions[index];
            }
        }
    }
    this->allow2TurnRight = allowRight;
    this->allow2TurnLeft = allowLeft;
}

void BlowRedLightDetector::getLightLocation(Point& plateLocation) {
    plateLocation = boundingRect(this->standardLight[0]).tl();
}

void BlowRedLightDetector::update(Mat& frame, vector<bbox_t> detected, vector<OutputData>& output) {
    try {
        Mat Frame;
        frame.copyTo(Frame);

        if ((this->standardLight.size() == 0)||(this->beginLine.size() == 0)||(this->endLine.size() == 0)||(this->leftLine.size() == 0)||(this->rightLine.size() == 0)) return;

        int widthFrame = frame.cols;
        int heightFrame = frame.rows;
        this->standardSignal = getCurrentLightState(frame, this->standardLight);
        this->leftSignal = getCurrentLightState(frame, this->leftLight);

        string status = getStatus(this->standardSignal, this->leftSignal);
        
        if (detected.size() != 0) {
            Point2f beginLinePtB((float) this->beginLine[0].x/widthFrame, (float) this->beginLine[0].y/heightFrame);
            Point2f beginLinePtE((float) this->beginLine[1].x/widthFrame, (float) this->beginLine[1].y/heightFrame);
            line(Frame, this->beginLine[0], this->beginLine[1], Scalar(0, 255, 0), 1, cv::LINE_AA);

            Point2f endLinePtB((float) this->endLine[0].x/widthFrame, (float) this->endLine[0].y/heightFrame);
            Point2f endLinePtE((float) this->endLine[1].x/widthFrame, (float) this->endLine[1].y/heightFrame);
            line(Frame, this->endLine[0], this->endLine[1], Scalar(0, 255, 0), 1, cv::LINE_AA);

            Mat detectFrame, trackingFrame;
            frame.copyTo(detectFrame);
            frame.copyTo(trackingFrame);

            vector<bbox_t> subDetect;
            for (auto &i: detected) {
                subDetect.push_back(i);
            }

            regions_t regions;
            bbox2regions(subDetect, regions, this->objNames);
            vector<TrackingObject> tracks;
            this->tracking->process(regions, frame, tracks, status);

            for (vehicle &ve: this->listVehicle) ve.isOutOfTheFrame = true;

            for (auto &track: tracks) {
                if (track.IsRobust(cvRound(3), 0.8f, Size2f(0.1f, 8.0f))) {
                    vehicle obj;
                    const int theId =  track.m_ID;
                    const auto p = find_if(this->listVehicle.begin(), this->listVehicle.end(), [theId] ( const vehicle& a ) { return a.id == theId;});
                    if (p != this->listVehicle.end()) {
                        int index = distance(this->listVehicle.begin(), p);
                        this->listVehicle[index].isOutOfTheFrame = false;
                        // if (this->listVehicle[index].box.size() <= 60) this->listVehicle[index].box.push_back(track.m_brect.boundingRect());
                        this->listVehicle[index].box.push_back(track.m_brect.boundingRect());

                        while (this->listVehicle[index].box.size() > 60) {
                            this->listVehicle[index].box.erase(this->listVehicle[index].box.begin());
                        }
                        // this->listVehicle[index].box.push_back(track.m_brect.boundingRect());
                        rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 255, 0), 1, LINE_AA);

                        if (!this->listVehicle[index].straight) line(Frame, track.m_brect.boundingRect().tl(), Point(track.m_brect.boundingRect().x + track.m_brect.boundingRect().width, track.m_brect.boundingRect().y), Scalar(0, 0, 255), 2, cv::LINE_AA);
                        if (!this->listVehicle[index].left) line(Frame, track.m_brect.boundingRect().tl(), Point(track.m_brect.boundingRect().x, track.m_brect.boundingRect().y + track.m_brect.boundingRect().height), Scalar(0, 0, 255), 2, cv::LINE_AA);
                        if (!this->listVehicle[index].right) line(Frame, Point(track.m_brect.boundingRect().x + track.m_brect.boundingRect().width, track.m_brect.boundingRect().y), track.m_brect.boundingRect().br(),  Scalar(0, 0, 255), 2, cv::LINE_AA);
                        if (checkStill(track)) this->listVehicle.erase(p);
                        if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(endLinePtB, endLinePtE, 0))) {
                            if (!this->listVehicle[index].straight) {
                                OutputData data;
                                data.id = this->listVehicle[index].id;
                                data.brect = this->listVehicle[index].box;
                                data.obj_id = convertType2Int(this->objNames, track.m_type);
                                output.push_back(data);
                                rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 0, 255), 3, LINE_AA);
                            }
                            this->listVehicle.erase(p);
                        }
                        if (this->isLeftLine) {
                            Point2f LeftLinePB((float) this->leftLine[0].x/widthFrame, (float) this->leftLine[0].y/heightFrame);
                            Point2f LeftLinePE((float) this->leftLine[1].x/widthFrame, (float) this->leftLine[1].y/heightFrame);
                            line(Frame, this->leftLine[0], this->leftLine[1], Scalar(0, 255, 0), 1, cv::LINE_AA);

                            if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(LeftLinePB, LeftLinePE, 0))) {
                                if (!listVehicle[index].left) {
                                    OutputData data;
                                    data.id = this->listVehicle[index].id;
                                    data.brect = this->listVehicle[index].box;
                                    data.obj_id = convertType2Int(this->objNames, track.m_type);
                                    output.push_back(data);
                                    rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 0, 255), 3, LINE_AA);
                                }
                                this->listVehicle.erase(p);
                            }
                        }
                        if (this->isRightLine) {
                            Point2f rightLinePB((float) this->rightLine[0].x/widthFrame, (float) this->rightLine[0].y/heightFrame);
                            Point2f rightLinePE((float) this->rightLine[1].x/widthFrame, (float) this->rightLine[1].y/heightFrame);
                            line(Frame, this->rightLine[0], this->rightLine[1], Scalar(0, 255, 0), 1, cv::LINE_AA);
                            if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(rightLinePB, rightLinePE, 0))) {
                                if (!listVehicle[index].right) {
                                    OutputData data;
                                    data.id = this->listVehicle[index].id;
                                    data.brect = this->listVehicle[index].box;
                                    data.obj_id = convertType2Int(this->objNames, track.m_type);
                                    output.push_back(data);
                                    rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 0, 255), 3, LINE_AA);
                                }
                                this->listVehicle.erase(p);
                            }
                        }
                    }
                    else {
                        if (tracking->checkIntersection(track, static_cast<float>(widthFrame), static_cast<float>(heightFrame), RoadLine(beginLinePtB, beginLinePtE, 0))) {
                            obj.id = track.m_ID;
                            obj.box.push_back(track.m_brect.boundingRect());
                            if (this->leftSignal == LightState::RED ||
                                (!allow2turn(this->allow2TurnLeft, track.m_type) && this->leftSignal == LightState::UNDEFINED && this->standardSignal == LightState::RED) ||
                                checkLane(track, this->straightLane) ||
                                checkLane(track, this->straight_rightlLane) ||
                                checkLane(track, this->rightLane)
                                ) obj.left = false;

                            if ((!allow2turn(this->allow2TurnRight, track.m_type) && this->standardSignal == LightState::RED) ||
                                checkLane(track, this->straightLane) ||
                                checkLane(track, this->straight_leftLane) ||
                                checkLane(track, this->leftLane)
                                ) obj.right = false;

                            if (this->standardSignal == LightState::RED ||
                                checkLane(track, this->rightLane) ||
                                checkLane(track, this->leftLane)
                                ) obj.straight = false;
                            obj.isOutOfTheFrame = false;
                            this->listVehicle.push_back(obj);
                            rectangle(Frame, track.m_brect.boundingRect(), Scalar(0, 255, 0), 3, LINE_AA);
                        }
                    }
                }
            }
            for (int i = 0; i < this->listVehicle.size(); i++) if (this->listVehicle[i].isOutOfTheFrame) this->listVehicle.erase(this->listVehicle.begin() + i);
        }
        imshow("Frame", Frame);
        waitKey(1);
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
}
bool BlowRedLightDetector::checkLane(TrackingObject track, vector<Point> area) {
    if (area.size() == 0) return false;
    Point2f cBox = Point2f(track.m_brect.boundingRect().x + float(track.m_brect.boundingRect().width/2), track.m_brect.boundingRect().y + float(track.m_brect.boundingRect().height/2));
    if (pointPolygonTest(area, cBox, false) >= 0) {
        return true;
    }
    return false;
}

float isBrighting(const Mat &input, Rect light) {
    Rect small;
    small.x = light.x + light.width/4;
    small.y = light.y + light.height/4;
    small.width = light.width - light.width/2;
    small.height = light.height - light.height/2;
    Mat roiImg = input(small);
    if (roiImg.empty()) return -1;
    Mat grey;
    cvtColor(roiImg, grey, COLOR_BGR2GRAY);
    return mean(grey)[0];
}

LightState BlowRedLightDetector::getCurrentLightState(const Mat &input, vector<vector<Point>> light) {
    if (light.size() == 0) return LightState::UNDEFINED;

    Rect red, yellow, green;
    float a,b,c;
    if (light.size() == 3) {
        red = boundingRect(light[0]);
        yellow = boundingRect(light[1]);
        green = boundingRect(light[2]);
        a = isBrighting(input, red);
        b = isBrighting(input, yellow);
        c = isBrighting(input, green);
    }
    if (light.size() == 2) {
        red = boundingRect(light[0]);
        green = boundingRect(light[1]);
        a = isBrighting(input, red);
        b = isBrighting(input, yellow);
        c = -1;
    }
    if (light.size() == 1) {
        red = boundingRect(light[0]);
        a = isBrighting(input, red);
        b = c = -1;
    }

    if (a < 100 && b < 100 && c < 100) return LightState::OFF;
    if (a > b && a > c) return LightState::RED;
    if (b > a && b > c) return LightState::YELLOW;
    if (c > a && c > b) return LightState::GREEN;
    return LightState::UNDEFINED;
}

bool BlowRedLightDetector::allow2turn(int allowTurn, string type) {
    switch (allowTurn)
    {
    case 0:
        return false;
        break;
    case 1:
        return true;
        break;
    case -1:
        return false;
        break;
    case 3:
        if (type == "xemay" || type == "xedap") {
            return true;
        }
        else {
            return false;
        }
        break;
    default:
        return false;
        break;
    }
}

string BlowRedLightDetector::getStatus(LightState standardSignal, LightState leftSignal) {
    string a, b;
    switch (standardSignal) {
        case LightState::GREEN:
            a = "G";
            break;
        case LightState::YELLOW:
            a = "Y";
            break;
        case LightState::RED:
            a = "R";
            break;
        case LightState::OFF:
        case LightState::UNDEFINED:
            a = "U";
            break;
        default:
            break;
    }
    switch (leftSignal) {
        case LightState::GREEN:
            b = "G";
            break;
        case LightState::YELLOW:
            b = "Y";
            break;
        case LightState::RED:
            b = "R";
            break;
        case LightState::OFF:
        case LightState::UNDEFINED:
            b = "U";
            break;
        default:
            break;
    }
    return a + "," + b;
}
bool BlowRedLightDetector::checkStill(TrackingObject tracker) {
    constexpr size_t minSize = 30;
    if (tracker.m_trace.size() > minSize) {
        Rect smallbox;
        smallbox.x = tracker.m_brect.boundingRect().x + tracker.m_brect.boundingRect().width*3/8;
        smallbox.y = tracker.m_brect.boundingRect().y + tracker.m_brect.boundingRect().height*3/8;
        smallbox.width = tracker.m_brect.boundingRect().width/4;
        smallbox.height = tracker.m_brect.boundingRect().height/4;

        const TrajectoryPoint &pt = tracker.m_trace.at(tracker.m_trace.size()-minSize);
        return smallbox.contains(Point(pt.m_prediction.x, pt.m_prediction.y));
    }
    return false;
}