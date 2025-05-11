#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

struct DetectionParams {
    int gaussian_size = 3;
    int threshold_value = 100;
    int min_contour_area = 50;
    int brightness_threshold = 200;  // 从100增加到200
    int bilateral_d = 7;
    double bilateral_sigma_color = 75;
    double bilateral_sigma_space = 75;
    int morph_open_size = 3;
    int dilate_iterations = 2;
    float min_aspect_ratio = 2.5f;   // 修改最小长宽比
    float max_aspect_ratio = 15.0f;  // 保持最大长宽比
};

class LightBarDetector {
private:
    DetectionParams params;
    Mat all_rect_debug;  // 添加用于显示所有矩形的调试图像
    struct LightBar {
        RotatedRect rect;
        Point2f top_point, bottom_point;
        float length, brightness;
        Vec3b color;

        LightBar() : brightness(0), color(Vec3b(0,0,0)), length(0) {}
        
        LightBar(const RotatedRect& r, float b, const Scalar& c) : brightness(b), color(Vec3b(c[0],c[1],c[2])) {
            rect = r;
            Point2f vertices[4];
            r.points(vertices);
            bool isVertical = r.size.width < r.size.height;
            top_point = (vertices[isVertical ? 0 : 0] + vertices[isVertical ? 3 : 1]) * 0.5f;
            bottom_point = (vertices[isVertical ? 1 : 2] + vertices[isVertical ? 2 : 3]) * 0.5f;
            length = norm(top_point - bottom_point);
        }
        LightBar(Point2f p1, Point2f p2) : top_point(p1), bottom_point(p2), length(norm(p1-p2)) {
            Point2f center = (p1 + p2) * 0.5f;
            rect = RotatedRect(center, Size2f(5.0f, length), atan2(p2.y-p1.y, p2.x-p1.x) * 180/CV_PI);
        }
    };

    struct ArmorPair { 
        LightBar left, right;
        ArmorPair(const LightBar& l={}, const LightBar& r={}) : left(l), right(r) {}
    };

    bool isSimilarPair(const ArmorPair& p1, const ArmorPair& p2, float threshold = 10.0f) {
        auto getRatio = [](const LightBar& l1, const LightBar& l2) {
            float ratio = norm(l1.rect.center - l2.rect.center) / ((l1.length + l2.length) * 0.5f);
            //cout << "灯条中心距/平均长度比值: " << ratio << endl;
            return ratio;
        };
        
        // 去掉这里的长宽比判断，因为在轮廓检测时已经判断过了
        float ratio1 = getRatio(p1.left, p1.right), ratio2 = getRatio(p2.left, p2.right);
        if(ratio1 < 1.0f || ratio1 > 3.5f || ratio2 < 1.0f || ratio2 > 3.5f) {
            //cout << "装甲板宽高比不合适: " << ratio1 << ", " << ratio2 << endl;
            return false;
        }
        
        auto getAngle = [](const Point2f& bottom, const Point2f& top) {
            return atan2(-(top.y - bottom.y), top.x - bottom.x); // 注意y轴向下为正
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
            //cout << "高度差比值过大: " << heightDiff1/avgLen1 << " (当前阈值: " << heightDiffThreshold 
            //     << ", 倾斜角度: " << angle_deg << "度)" << endl;
            return false;
        }
        
        float angle1 = atan2(p1.right.rect.center.y - p1.left.rect.center.y, 
                            p1.right.rect.center.x - p1.left.rect.center.x);
        float angle2 = atan2(p2.right.rect.center.y - p2.left.rect.center.y, 
                            p2.right.rect.center.x - p2.left.rect.center.x);
        if(abs(angle1 - angle2) > CV_PI/6) {
            //cout << "角度差过大: " << abs(angle1 - angle2)*180/CV_PI << "度 (应小于30度)" << endl;
            return false;
        }
        
        Point2f center1 = p1.left.rect.center;
        Point2f center2 = p2.left.rect.center;
        return norm(center1 - center2) < threshold;
    }

    Mat getBrightnessFilteredImage(const Mat& gray) {
        Mat brightMask = Mat::zeros(gray.size(), CV_8UC1);
        threshold(gray, brightMask, params.brightness_threshold, 255, THRESH_BINARY);
        return brightMask;
    }

    Mat getAreaFilteredImage(const Mat& binary) {
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

    void showPreprocessing(const Mat& frame, const Mat& gray, const Mat& binary) {
        // 创建一个3x2的网格显示
        int width = frame.cols/3;
        int height = frame.rows/2;
        Mat display(height*2, width*3, CV_8UC3);
        display = Scalar(0,0,0);
        
        // 原始图像
        Mat roi = display(Rect(0, 0, width, height));
        resize(frame, roi, roi.size());
        putText(roi, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        
        // 灰度图
        Mat gray_color;
        cvtColor(gray, gray_color, COLOR_GRAY2BGR);
        roi = display(Rect(width, 0, width, height));
        resize(gray_color, roi, roi.size());
        putText(roi, "Grayscale", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        
        // 二值化结果
        Mat bin_color;
        cvtColor(binary, bin_color, COLOR_GRAY2BGR);
        roi = display(Rect(width*2, 0, width, height));
        resize(bin_color, roi, roi.size());
        putText(roi, "Binary", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        
        // 亮度筛选结果
        Mat bright_mask = getBrightnessFilteredImage(gray);
        Mat bright_color;
        cvtColor(bright_mask, bright_color, COLOR_GRAY2BGR);
        roi = display(Rect(0, height, width, height));
        resize(bright_color, roi, roi.size());
        putText(roi, "Brightness Filter", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        
        // 面积筛选结果
        Mat area_mask = getAreaFilteredImage(binary);
        Mat area_color;
        cvtColor(area_mask, area_color, COLOR_GRAY2BGR);
        roi = display(Rect(width, height, width, height));
        resize(area_color, roi, roi.size());
        putText(roi, "Area Filter", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        putText(roi, "Min Area: " + to_string(params.min_contour_area), 
                Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
        
        // 最终结果（面积+亮度筛选）
        Mat final_mask = area_mask & bright_mask;
        Mat final_color;
        cvtColor(final_mask, final_color, COLOR_GRAY2BGR);
        roi = display(Rect(width*2, height, width, height));
        resize(final_color, roi, roi.size());
        putText(roi, "Final Result", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        
        imshow("Preprocessing Steps", display);
    }

public:
    Mat debug_img;
    vector<ArmorPair> matchLightBars(const vector<LightBar>& lights) {
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

    void showDebugInfo(const Mat& frame, const vector<LightBar>& lights) {
        debug_img = frame.clone();
        auto pairs = matchLightBars(lights);
        
        for(const auto& p : pairs) {
            // 获取所有角点
            vector<Point2f> corners = {
                p.left.top_point, p.right.top_point,
                p.right.bottom_point, p.left.bottom_point
            };
            
            // 按照左上,右上,右下,左下排序
            sort(corners.begin(), corners.end(), 
                [](const Point2f& p1, const Point2f& p2) {
                    return p1.x + p1.y < p2.x + p2.y;
                });
            
            // 绘制装甲板四边形
            for(int i = 0; i < 4; i++) {
                line(debug_img, corners[i], corners[(i+1)%4], Scalar(0,255,255), 2);
            }
            
            // 中心点
            Point2f center = (p.left.rect.center + p.right.rect.center) * 0.5f;
            circle(debug_img, center, 3, Scalar(0,0,255), -1);
        }
        
        imshow("Debug Window", debug_img);
    }

    vector<LightBar> detect(Mat& frame) {
        auto lights = vector<LightBar>();
        all_rect_debug = frame.clone();  // 初始化调试图像
        debug_img = frame.clone();  // 移动到函数开始处，用于绘制所有矩形
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 1. 亮度筛选
        Mat brightMask = getBrightnessFilteredImage(gray);
        
        // 2. 二值化 
        Mat bin;
        threshold(gray, bin, params.threshold_value, 255, THRESH_BINARY);
        bin = bin & brightMask;  // 应用亮度mask
        
        // 3. 面积筛选
        Mat areaMask = getAreaFilteredImage(bin);
        bin = bin & areaMask;
        
        // 4. 膨胀操作，连接可能断开的灯条
        Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(bin, bin, element);
        
        // 显示预处理过程
        showPreprocessing(frame, gray, bin);
        
        // 使用预处理结果查找轮廓
        vector<vector<Point>> contours;
        findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        //cout << "找到 " << contours.size() << " 个轮廓" << endl;
        
        for(const auto& c : contours) {
            try {
                RotatedRect r = minAreaRect(c);
                Point2f vertices[4];
                r.points(vertices);

                // 计算最小外接矩形的长宽比
                float long_side = max(r.size.width, r.size.height);
                float short_side = min(r.size.width, r.size.height);
                float aspect_ratio = long_side / short_side;
                
                //cout << "轮廓的最小外接矩形 - 长边: " << long_side 
                //     << " 短边: " << short_side
                //     << " 长宽比: " << aspect_ratio << endl;
                
                // 只要长宽比符合要求,就认为是灯带
                if(aspect_ratio >= params.min_aspect_ratio && aspect_ratio <= params.max_aspect_ratio) {
                    // 绘制找到的灯带
                    for(int i = 0; i < 4; i++) {
                        line(all_rect_debug, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 2);
                    }
                    lights.emplace_back(r, mean(gray(r.boundingRect() & Rect(0,0,frame.cols,frame.rows)))[0], mean(frame(r.boundingRect())));
                    cout << "找到一个灯带!" << endl;
                }
            } catch(...) {
                cout << "处理轮廓时出现异常" << endl;
            }
        }
        
        cout << "最终找到 " << lights.size() << " 个灯条" << endl;
        showDebugInfo(frame, lights);
        imshow("All Rectangles", all_rect_debug);  // 显示所有矩形的调试窗口
        return lights;
    }
};

int main() {
    VideoCapture cap("vedio.mp4");
    if(!cap.isOpened()) {
        cout << "无法打开视频文件！" << endl;
        return -1;
    }

    LightBarDetector detector;
    Mat frame;
    
    namedWindow("Debug Window", WINDOW_AUTOSIZE);  // 只保留debug窗口
    namedWindow("Preprocessing Steps", WINDOW_AUTOSIZE);  // 添加预处理窗口
    namedWindow("All Rectangles", WINDOW_AUTOSIZE);  // 添加新的窗口
    
    while(true) {
        if(!cap.read(frame)) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        detector.detect(frame);
        
        char key = waitKey();  // 降低延迟，提高显示帧率
        if(key == 27) break;
        else if(key == 's') {
            imwrite("light_bars.jpg", detector.debug_img);
        }
    }
    
    destroyAllWindows();
    return 0;
}