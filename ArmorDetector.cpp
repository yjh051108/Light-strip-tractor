#include "ArmorDetector.h"
#include <iostream>

using namespace cv;
using namespace std;

ArmorDetector::LightBar::LightBar() : brightness(0), color(Vec3b(0,0,0)), length(0) {}

ArmorDetector::LightBar::LightBar(const RotatedRect& r, float b, const Scalar& c) : 
    brightness(b), color(Vec3b(c[0],c[1],c[2])) {
    rect = r;
    Point2f vertices[4];
    r.points(vertices);
    bool isVertical = r.size.width < r.size.height;
    top_point = (vertices[isVertical ? 0 : 0] + vertices[isVertical ? 3 : 1]) * 0.5f;
    bottom_point = (vertices[isVertical ? 1 : 2] + vertices[isVertical ? 2 : 3]) * 0.5f;
    length = norm(top_point - bottom_point);
}

ArmorDetector::LightBar::LightBar(Point2f p1, Point2f p2) : 
    top_point(p1), bottom_point(p2), length(norm(p1-p2)) {
    Point2f center = (p1 + p2) * 0.5f;
    rect = RotatedRect(center, Size2f(5.0f, length), atan2(p2.y-p1.y, p2.x-p1.x) * 180/CV_PI);
}

ArmorDetector::ArmorPair::ArmorPair(const LightBar& l, const LightBar& r) : left(l), right(r) {}

bool ArmorDetector::isSimilarPair(const ArmorPair& p1, const ArmorPair& p2, float threshold) {
    auto getRatio = [](const LightBar& l1, const LightBar& l2) {
        return norm(l1.rect.center - l2.rect.center) / ((l1.length + l2.length) * 0.5f);
    };
    
    float ratio1 = getRatio(p1.left, p1.right), ratio2 = getRatio(p2.left, p2.right);
    if(ratio1 < 1.0f || ratio1 > 3.5f || ratio2 < 1.0f || ratio2 > 3.5f) {
        return false;
    }
    
    auto getAngle = [](const Point2f& bottom, const Point2f& top) {
        return atan2(-(top.y - bottom.y), top.x - bottom.x);
    };
    
    float angle_left = abs(getAngle(p1.left.bottom_point, p1.left.top_point));
    float angle_right = abs(getAngle(p1.right.bottom_point, p1.right.top_point));
    float avg_angle = (angle_left + angle_right) * 0.5f;
    float angle_deg = avg_angle * 180.0f / CV_PI;
    
    float heightDiffThreshold = 2.5f + (min(abs(angle_deg), 30.0f) / 30.0f) * 0.5f;
    float heightDiff1 = abs(p1.left.rect.center.y - p1.right.rect.center.y);
    float heightDiff2 = abs(p2.left.rect.center.y - p2.right.rect.center.y);
    float avgLen1 = (p1.left.length + p1.right.length) * 0.5f;
    float avgLen2 = (p2.left.length + p2.right.length) * 0.5f;
    if(heightDiff1 > heightDiffThreshold * avgLen1 || heightDiff2 > heightDiffThreshold * avgLen2) {
        return false;
    }
    
    float angle1 = atan2(p1.right.rect.center.y - p1.left.rect.center.y, 
                        p1.right.rect.center.x - p1.left.rect.center.x);
    float angle2 = atan2(p2.right.rect.center.y - p2.left.rect.center.y, 
                        p2.right.rect.center.x - p2.left.rect.center.x);
    if(abs(angle1 - angle2) > CV_PI/6) {
        return false;
    }
    
    Point2f center1 = p1.left.rect.center;
    Point2f center2 = p2.left.rect.center;
    return norm(center1 - center2) < threshold;
}

Mat ArmorDetector::getBrightnessFilteredImage(const Mat& gray) {
    Mat brightMask = Mat::zeros(gray.size(), CV_8UC1);
    threshold(gray, brightMask, params.brightness_threshold, 255, THRESH_BINARY);
    return brightMask;
}

Mat ArmorDetector::getAreaFilteredImage(const Mat& binary) {
    Mat areaMask = Mat::zeros(binary.size(), CV_8UC1);
    vector<vector<Point>> contours;
    findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for(const auto& contour : contours) {
        if(contourArea(contour) >= params.min_contour_area) {
            drawContours(areaMask, vector<vector<Point>>{contour}, 0, Scalar(255), -1);
        }
    }
    return areaMask;
}

vector<ArmorDetector::ArmorPair> ArmorDetector::matchLightBars(const vector<LightBar>& lights) {
    vector<ArmorPair> pairs;
    vector<bool> used(lights.size());
    
    #pragma omp parallel for
    for(size_t i = 0; i < lights.size(); i++) {
        if(used[i]) continue;
        for(size_t j = i + 1; j < lights.size(); j++) {
            if(used[j]) continue;
            ArmorPair pair(lights[i], lights[j]);
            if(isSimilarPair(pair, pair, 200.0f)) {
                #pragma omp critical
                {
                    if(!used[i] && !used[j]) {
                        pairs.push_back(pair);
                        used[i] = used[j] = true;
                    }
                }
            }
        }
    }
    return pairs;
}

void ArmorDetector::showDebugInfo(const Mat& frame, const vector<LightBar>& lights) {
    debug_img = frame.clone();
    auto pairs = matchLightBars(lights);
    
    for(const auto& p : pairs) {
        vector<Point2f> corners = {
            p.left.top_point, p.right.top_point,
            p.right.bottom_point, p.left.bottom_point
        };
        
        sort(corners.begin(), corners.end(), 
            [](const Point2f& p1, const Point2f& p2) {
                return p1.x + p1.y < p2.x + p2.y;
            });
        
        for(int i = 0; i < 4; i++) {
            line(debug_img, corners[i], corners[(i+1)%4], Scalar(0,255,255), 2);
        }
        
        Point2f center = (p.left.rect.center + p.right.rect.center) * 0.5f;
        circle(debug_img, center, 3, Scalar(0,0,255), -1);
    }
    
    imshow("Debug Window", debug_img);
}

vector<Point2f> ArmorDetector::detectArmorCenters(const Mat& frame) {
    vector<LightBar> lights;
    all_rect_debug = frame.clone();
    debug_img = frame.clone();
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    Mat brightMask = getBrightnessFilteredImage(gray);
    Mat bin;
    threshold(gray, bin, params.threshold_value, 255, THRESH_BINARY);
    bin = bin & brightMask;
    
    Mat areaMask = getAreaFilteredImage(bin);
    bin = bin & areaMask;
    
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    dilate(bin, bin, element);
    
    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for(const auto& c : contours) {
        try {
            RotatedRect r = minAreaRect(c);
            float long_side = max(r.size.width, r.size.height);
            float short_side = min(r.size.width, r.size.height);
            float aspect_ratio = long_side / short_side;
            
            if(aspect_ratio >= params.min_aspect_ratio && aspect_ratio <= params.max_aspect_ratio) {
                Rect bbox = r.boundingRect();
                if(bbox.x >= 0 && bbox.y >= 0 && 
                   bbox.x + bbox.width <= frame.cols && 
                   bbox.y + bbox.height <= frame.rows) {
                    lights.emplace_back(r, mean(gray(bbox))[0], mean(frame(bbox)));
                }
            }
        } catch(...) {}
    }
    
    showDebugInfo(frame, lights);
    imshow("All Rectangles", all_rect_debug);
    
    vector<Point2f> centers;
    auto pairs = matchLightBars(lights);
    for(const auto& p : pairs) {
        centers.push_back((p.left.rect.center + p.right.rect.center) * 0.5f);
    }
    
    return centers;
}