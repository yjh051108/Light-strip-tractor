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
    Mat prev_frame;             // 保存上一帧
    
    static constexpr int HISTORY_FRAME_COUNT = 120;  // 修改为120帧
    static constexpr float UNBIND_THRESHOLD = 0.9f; // 解绑阈值
    static constexpr float MOTION_DEVIATION_THRESHOLD = 0.5f;  // 运动解离阈值(需要明显偏离)
    static constexpr float MOTION_SIMILARITY_THRESHOLD = 0.99f; // 运动相似度阈值(需要非常相似)
    
    struct LightBar {
        RotatedRect rect;
        float brightness;
        Vec3b color;
        Point2f top_point;    // 添加端点
        Point2f bottom_point; // 添加端点
        
        // 原有构造函数
        LightBar(const RotatedRect& r, float b, const Scalar& c) 
            : rect(r), brightness(b), color(Vec3b(c[0], c[1], c[2])) {}
        
        // 添加新的构造函数
        LightBar(Point2f p1, Point2f p2) 
            : top_point(p1), bottom_point(p2), 
              brightness(0), color(Vec3b(0,0,0)) {
            Point2f center = (p1 + p2) * 0.5f;
            float angle = atan2(p2.y - p1.y, p2.x - p1.x);
            Size2f size(5.0f, norm(p2 - p1)); // 设置一个默认宽度
            rect = RotatedRect(center, size, angle * 180 / CV_PI);
        }
    };





public:
    Mat debug_img;
    
    // 修改匹配函数


    void showDebugInfo(const Mat& frame, const vector<LightBar>& lights) {
        debug_img = frame.clone();
        
        // 显示灯条端点
        for(const auto& light : lights) {
            Point2f vertices[4];
            light.rect.points(vertices);
            
            // 确定短边并计算中点
            float width = light.rect.size.width;
            float height = light.rect.size.height;
            
            // 短边是宽度或高度中较小的那个
            if(width < height) {
                // 短边是宽度，取顶点0和3的中点，顶点1和2的中点
                Point2f p1 = (vertices[0] + vertices[3]) * 0.5;
                Point2f p2 = (vertices[1] + vertices[2]) * 0.5;
                circle(debug_img, p1, 3, Scalar(0,255,255), FILLED);
                circle(debug_img, p2, 3, Scalar(0,255,255), FILLED);
            } else {
                // 短边是高度，取顶点0和1的中点，顶点2和3的中点
                Point2f p1 = (vertices[0] + vertices[1]) * 0.5;
                Point2f p2 = (vertices[2] + vertices[3]) * 0.5;
                circle(debug_img, p1, 3, Scalar(0,255,255), FILLED);
                circle(debug_img, p2, 3, Scalar(0,255,255), FILLED);
            }
        }
        
        imshow("Debug Window", debug_img);
    }
    
    vector<LightBar> detect(Mat& frame) {
        double start_time = static_cast<double>(getTickCount());
        
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
        
     
        
        // 显示调试信息
           showDebugInfo(frame, lights);
        
        double end_time = static_cast<double>(getTickCount());
        double elapsed_time = (end_time - start_time) / getTickFrequency();
        cout << "单次识别耗时: " << elapsed_time * 1000 << "ms" << endl;
        
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