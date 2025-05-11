#include <opencv2/opencv.hpp> 
#include <iostream>
#include <vector>
#include <omp.h>
using namespace cv;
using namespace std;

// 参数结构体
struct DetectionParams {
    int gaussian_size = 3;          // 高斯模糊核大小
    int threshold_value = 100;      // 二值化阈值
    int min_contour_area = 50;      // 最小轮廓面积
    int brightness_threshold = 58;  // 亮度阈值
    int bilateral_d = 7;            // 双边滤波直径
    double bilateral_sigma_color = 75; // 双边滤波颜色空间标准差
    double bilateral_sigma_space = 75; // 双边滤波坐标空间标准差
    int morph_open_size = 3;        // 形态学开操作核大小
    int dilate_iterations = 2;      // 膨胀迭代次数
};

// 修改ArmorDetector类，移除运动轨迹相关成员
class LightBarDetector {
private:
    DetectionParams params;

    
    static constexpr int HISTORY_FRAME_COUNT = 180;  // 修改为180帧
    static constexpr float UNBIND_THRESHOLD = 0.95f;
    static constexpr float MOTION_DEVIATION_THRESHOLD = 0.5f;
    static constexpr float MOTION_SIMILARITY_THRESHOLD = 0.998f;
    
    struct LightBar {
        RotatedRect rect;
        float brightness;
        Vec3b color;
        Point2f top_point;    // 添加端点
        Point2f bottom_point; // 添加端点
        float length;         // 添加灯带长度
        
        LightBar(const RotatedRect& r, float b, const Scalar& c) 
            : rect(r), brightness(b), color(Vec3b(c[0], c[1], c[2])) {
            Point2f vertices[4];
            r.points(vertices);
            float width = r.size.width;
            float height = r.size.height;
            
            if(width < height) {
                top_point = (vertices[0] + vertices[3]) * 0.5f;
                bottom_point = (vertices[1] + vertices[2]) * 0.5f;
            } else {
                top_point = (vertices[0] + vertices[1]) * 0.5f;
                bottom_point = (vertices[2] + vertices[3]) * 0.5f;
            }
            length = norm(top_point - bottom_point);
        }
            
        LightBar(Point2f p1, Point2f p2) 
            : top_point(p1), bottom_point(p2), 
              brightness(0), color(Vec3b(0,0,0)) {
            Point2f center = (p1 + p2) * 0.5f;
            float angle = atan2(p2.y - p1.y, p2.x - p1.x);
            Size2f size(5.0f, norm(p2 - p1)); // 设置一个默认宽度
            rect = RotatedRect(center, size, angle * 180 / CV_PI);
            length = norm(p1 - p2);
        }
    };

    struct ArmorPair {
        LightBar left;
        LightBar right;
        ArmorPair() : left(LightBar(Point2f(0, 0), Point2f(0, 0))), right(LightBar(Point2f(0, 0), Point2f(0, 0))) {}
        ArmorPair(const LightBar& l, const LightBar& r) : left(l), right(r) {}
    };

    // 修改 isSimilarPair 函数
    bool isSimilarPair(const ArmorPair& p1, const ArmorPair& p2, float threshold = 10.0f) {
        // 计算中点连线长度
        float connectLen1 = norm(p1.left.rect.center - p1.right.rect.center);
        float connectLen2 = norm(p2.left.rect.center - p2.right.rect.center);
        
        // 严格检查中点连线长度与灯带长度的比值
        float leftLen1 = norm(p1.left.top_point - p1.left.bottom_point);
        float rightLen1 = norm(p1.right.top_point - p1.right.bottom_point);
        float leftLen2 = norm(p2.left.top_point - p2.left.bottom_point);
        float rightLen2 = norm(p2.right.top_point - p2.right.bottom_point);
        
        float ratio1 = connectLen1 / ((leftLen1 + rightLen1)/2);
        float ratio2 = connectLen2 / ((leftLen2 + rightLen2)/2);
        if(ratio1 < 1.5f || ratio1 > 2.8f || ratio2 < 1.5f || ratio2 > 2.8f) {
            return false;
        }
        
        // 计算两个灯带与Y轴的平均倾斜角度（取绝对值）
        float angle_left = abs(atan2(p1.left.bottom_point.y - p1.left.top_point.y, 
                                   p1.left.bottom_point.x - p1.left.top_point.x));
        float angle_right = abs(atan2(p1.right.bottom_point.y - p1.right.top_point.y, 
                                    p1.right.bottom_point.x - p1.right.top_point.x));
        float avg_angle = (angle_left + angle_right) * 0.5f;
        float angle_deg = avg_angle * 180.0f / CV_PI;
        
        // 根据角度计算动态高度差阈值（线性插值）
        // 角度从0到15度，阈值从1.5到2.0线性变化
        float heightDiffThreshold = 1.5f + (min(angle_deg, 15.0f) / 15.0f) * 0.5f;
        
        // 使用动态阈值判断高度差
        float heightDiff1 = abs(p1.left.rect.center.y - p1.right.rect.center.y);
        float heightDiff2 = abs(p2.left.rect.center.y - p2.right.rect.center.y);
        float avgLen1 = (leftLen1 + rightLen1) * 0.5f;
        float avgLen2 = (leftLen2 + rightLen2) * 0.5f;
        if(heightDiff1 > heightDiffThreshold * avgLen1 || heightDiff2 > heightDiffThreshold * avgLen2) {
            cout << "高度差比值过大: " << heightDiff1/avgLen1 << " (当前阈值: " << heightDiffThreshold 
                 << ", 倾斜角度: " << angle_deg << "度)" << endl;
            return false;
        }
        
        // 角度差不超过15度
        float angle1 = atan2(p1.right.rect.center.y - p1.left.rect.center.y, 
                            p1.right.rect.center.x - p1.left.rect.center.x);
        float angle2 = atan2(p2.right.rect.center.y - p2.left.rect.center.y, 
                            p2.right.rect.center.x - p2.left.rect.center.x);
        if(abs(angle1 - angle2) > CV_PI/6) { // 30度
            cout << "角度差过大: " << abs(angle1 - angle2)*180/CV_PI << "度 (应小于30度)" << endl;
            return false;
        }
        
        // 中点连线角度不超过60度
        float midAngle1 = atan2(p1.right.rect.center.y - p1.left.rect.center.y, 
                               p1.right.rect.center.x - p1.left.rect.center.x);
        float midAngle2 = atan2(p2.right.rect.center.y - p2.left.rect.center.y, 
                               p2.right.rect.center.x - p2.left.rect.center.x);
        if((midAngle1 > -CV_PI*2/3 && midAngle1 < -CV_PI/3) || (midAngle1 > CV_PI/3 && midAngle1 < CV_PI*2/3) || 
           (midAngle2 > -CV_PI*2/3 && midAngle2 < -CV_PI/3) || (midAngle2 > CV_PI/3 && midAngle2 < CV_PI*2/3)) { // -120到-180度和120到180度
            cout << "中点连线角度过大: " << midAngle1*180/CV_PI << "度和" << midAngle2*180/CV_PI << "度 (应在-180到-120度或120到180度之间)" << endl;
            return false;
        }
        

        
        // 中心点距离阈值
        Point2f center1 = p1.left.rect.center;
        Point2f center2 = p2.left.rect.center;
        return norm(center1 - center2) < threshold;
    }

    // 判断是否为相同的绑定方案
    bool isSameBinding(const ArmorPair& p1, const ArmorPair& p2, float threshold = 10.0f) {
        return isSimilarPair(p1, p2) && 
               norm(p1.left.rect.center - p2.left.rect.center) < threshold &&
               norm(p1.right.rect.center - p2.right.rect.center) < threshold;
    }

public:
    Mat debug_img;
    
    // 修改匹配函数
    vector<ArmorPair> matchLightBars(const vector<LightBar>& lights) {
        double pair_start_time = static_cast<double>(getTickCount());
        vector<ArmorPair> curr_pairs;
        vector<bool> used(lights.size(), false);

        // 简单匹配灯条
        for(size_t i = 0; i < lights.size(); i++) {
            if(used[i]) continue;
            for(size_t j = i + 1; j < lights.size(); j++) {
                if(used[j]) continue;
                
                Point2f center_i = lights[i].rect.center;
                Point2f center_j = lights[j].rect.center;
                float dist = norm(center_i - center_j);
                
                ArmorPair new_pair(lights[i], lights[j]);
                    if(isSimilarPair(new_pair, new_pair, 200.0f)) {
                        curr_pairs.push_back(new_pair);
                        used[i] = used[j] = true;
                        break;
                    } else {
                        float leftLen1 = norm(new_pair.left.top_point - new_pair.left.bottom_point);
                        float rightLen1 = norm(new_pair.right.top_point - new_pair.right.bottom_point);
                        float connectLen = norm(new_pair.left.rect.center - new_pair.right.rect.center);
                        float ratio = connectLen / ((leftLen1 + rightLen1)/2);
                        float heightDiffRatio = abs(new_pair.left.rect.center.y - new_pair.right.rect.center.y) / ((leftLen1 + rightLen1)/2);
                        float angle1 = atan2(new_pair.right.rect.center.y - new_pair.left.rect.center.y, 
                                            new_pair.right.rect.center.x - new_pair.left.rect.center.x);
                        
                        if(ratio < 1.0f || ratio > 3.3f) {
                            // 不输出已知的正确阻拦情况
                        }
                        if(heightDiffRatio > 1.5f) {
                            cout << "高度差比值过大: " << heightDiffRatio << " (应小于1.5)" << endl;
                        }
                        if(abs(leftLen1 - rightLen1) > 0.4f * (leftLen1 + rightLen1)/2) {
                            cout << "灯带长度差异过大: " << abs(leftLen1 - rightLen1)/((leftLen1 + rightLen1)/2) << " (应小于0.4)" << endl;
                        }
                    }
            }
        }

        double pair_time = (getTickCount() - pair_start_time) / getTickFrequency();
        // 仅输出匹配失败时的调试信息
        
        return curr_pairs;
    }

    void showDebugInfo(const Mat& frame, const vector<LightBar>& lights) {
        debug_img = frame.clone();
        
        // 匹配装甲板
        auto armor_pairs = matchLightBars(lights);
        
        // 绘制装甲板所有端点之间的连线
        for(const auto& pair : armor_pairs) {
            vector<Point2f> points = {
                pair.left.top_point,
                pair.left.bottom_point,
                pair.right.top_point,
                pair.right.bottom_point
            };
            
            // 任意两点之间都画线
            for(int i = 0; i < 4; i++) {
                for(int j = i + 1; j < 4; j++) {
                    line(debug_img, points[i], points[j], Scalar(0,255,0), 2);
                }
            }
        }
        
        imshow("Debug Window", debug_img);
    }
    
    vector<LightBar> detect(Mat& frame) {
        double total_start_time = static_cast<double>(getTickCount());
        
        Mat gray, bin, gaussian_blur;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 应用双边滤波
        bilateralFilter(gray, gaussian_blur, params.bilateral_d, params.bilateral_sigma_color, params.bilateral_sigma_space);
        
        // 形态学处理（优化参数以增强灯条轮廓）
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(params.morph_open_size, params.morph_open_size));
        morphologyEx(gaussian_blur, gray, MORPH_OPEN, element);
        // 调整膨胀次数以增强灯条连接
        dilate(gray, gray, element, Point(-1,-1), params.dilate_iterations);
        
        // 应用二值化
        threshold(gray, bin, params.threshold_value, 255, THRESH_BINARY);
        
        vector<vector<Point>> contours;
        findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        vector<LightBar> lights;
        vector<bool> matched(contours.size(), false);
        
        for(size_t i = 0; i < contours.size(); i++) {
            if(matched[i]) continue;
            auto& c = contours[i];
            
            // 检查点数是否足够
            if(c.size() < 4) continue;
            
            try {
                RotatedRect r = minAreaRect(c);
                // 获取边界框并确保在图像范围内
                Rect boundRect = r.boundingRect() & Rect(0, 0, gray.cols, gray.rows);
                if(boundRect.width <= 0 || boundRect.height <= 0) continue;
                Scalar avgColor = mean(frame(boundRect));
                float brightness = mean(gray(boundRect))[0];
                // 检查长宽比和颜色条件
                float aspect_ratio = max(r.size.width, r.size.height) / min(r.size.width, r.size.height);
                if(avgColor[0] < avgColor[2] && avgColor[1] < avgColor[2] && 
                   (brightness > params.brightness_threshold || aspect_ratio > 2.5)) {
                    
                    // 直接添加检测到的灯条
                    lights.emplace_back(r, mean(gray(boundRect))[0], avgColor);
                    matched[i] = true;
                }
            } catch(cv::Exception& e) {
                continue;
            }
        }
        
        vector<LightBar> light_bars;
        for(const auto& r : lights) {
            Point2f vertices[4];
            r.rect.points(vertices);
            float width = r.rect.size.width;
            float height = r.rect.size.height;
            
            if(width < height) {
                light_bars.emplace_back(
                    (vertices[0] + vertices[3]) * 0.5f,
                    (vertices[1] + vertices[2]) * 0.5f
                );
            } else {
                light_bars.emplace_back(
                    (vertices[0] + vertices[1]) * 0.5f,
                    (vertices[2] + vertices[3]) * 0.5f
                );
            }
        }
        
        // 匹配灯条
        auto armor_pairs = matchLightBars(lights);
        
        double total_end_time = static_cast<double>(getTickCount());
        double total_elapsed_time = (total_end_time - total_start_time) / getTickFrequency();
        string total_time_text = "总耗时: " + to_string(total_elapsed_time * 1000) + "ms (包含匹配)";
        putText(frame, total_time_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        
        // 显示调试信息
        showDebugInfo(frame, lights);
        
        return lights;
    }
};

int main() {
    VideoCapture cap("2.mp4");
    if(!cap.isOpened()) {
        cout << "无法打开视频文件！" << endl;
        return -1;
    }

    LightBarDetector detector;
    Mat frame;
    
    // 创建窗口
    namedWindow("Debug Window", WINDOW_AUTOSIZE);
    
    while(true) {
        if(!cap.read(frame)) {
            cap.set(CAP_PROP_POS_FRAMES, 0); // 视频结束后重置到开头
            continue;
        }

        auto lights = detector.detect(frame); 
        
        char key = waitKey(40); // 增加延迟时间到40ms，实现0.5倍速播放
        if(key == 27) break;  // ESC退出
        else if(key == 's') {  // 按's'保存当前帧
            imwrite("light_bars.jpg", detector.debug_img);
        }
    }
    
    destroyAllWindows();
    return 0;
}