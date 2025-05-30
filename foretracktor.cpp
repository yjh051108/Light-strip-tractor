#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "MotionEstimator.h"
using namespace cv;
using namespace std;

struct DetectionParams {
    int gaussian_size = 3;
    int threshold_value = 100;
    int min_contour_area = 15;
    int brightness_threshold = 100;  // 从100增加到200
    int bilateral_d = 7;
    double bilateral_sigma_color = 75;
    double bilateral_sigma_space = 75;
    int morph_open_size = 3;
    int dilate_iterations = 2;
    float min_aspect_ratio = 1.5f;   // 修改最小长宽比
    float max_aspect_ratio = 100.0f;  // 保持最大长宽比
};

class ExtendedKalmanFilter {
private:
    Mat state;         // 状态向量 [x, y, vt, vn, at, an, jt, jn]
    Mat P;             // 状态协方差矩阵
    Mat Q;             // 过程噪声协方差
    Mat R;             // 测量噪声协方差
    double dt;         // 时间步长
    int last_frame_id;      // 记录上一帧ID
    Point2f last_tangential;  // 上一次的切向量
    Point2f last_normal;     // 上一次的法向量

    // 计算切向和法向单位向量
    void computeDirections(const Point2f& vel, Point2f& tangential, Point2f& normal) {
        float vel_norm = norm(vel);
        if(vel_norm < 1e-6) {
            // 如果速度太小,使用上一次的方向
            tangential = last_tangential;
            normal = last_normal;
            return;
        }
        tangential = vel / vel_norm;
        normal = Point2f(-tangential.y, tangential.x); // 逆时针旋转90度得到法向
        
        last_tangential = tangential;
        last_normal = normal;
    }

public:
    ExtendedKalmanFilter() {
        state = Mat::zeros(8, 1, CV_64F);
        P = Mat::eye(8, 8, CV_64F) * 100;
        // 调整过程噪声：切向速度噪声较小,法向加速度噪声较大
        Q = Mat::eye(8, 8, CV_64F);
        Q.at<double>(2,2) *= 0.1;  // 切向速度
        Q.at<double>(3,3) *= 0.5;  // 法向速度
        Q.at<double>(4,4) *= 0.2;  // 切向加速度
        Q.at<double>(5,5) *= 1.0;  // 法向加速度
        Q.at<double>(6,6) *= 0.05; // 切向加加速度
        Q.at<double>(7,7) *= 0.1;  // 法向加加速度
        R = Mat::eye(2, 2, CV_64F) * 1.0;
        dt = 1.0/30.0;
        last_frame_id = -1;  // 初始化为无效值
        last_tangential = Point2f(1, 0);
        last_normal = Point2f(0, 1);
    }

    void init(const Point2f& pos, const Point2f& vel = Point2f(0, 0)) {
        Point2f tangential, normal;
        computeDirections(vel, tangential, normal);
        
        float vel_mag = norm(vel);
        float vt = vel_mag * tangential.dot(vel/vel_mag);
        float vn = vel_mag * normal.dot(vel/vel_mag);

        state.at<double>(0) = pos.x;        // x位置
        state.at<double>(1) = pos.y;        // y位置
        state.at<double>(2) = vt;           // 切向速度
        state.at<double>(3) = vn;           // 法向速度
        state.at<double>(4) = 0;            // 切向加速度
        state.at<double>(5) = 0;            // 法向加速度
        state.at<double>(6) = 0;            // 切向加加速度
        state.at<double>(7) = 0;            // 法向加加速度
        state.at<double>(6) = 0;            // 切向加加速度
        state.at<double>(7) = 0;            // 法向加加速度
    }

    Mat f(const Mat& x) {
        Mat result = Mat::zeros(8, 1, CV_64F);
        double dt2 = dt * dt / 2.0;
        double dt3 = dt * dt * dt / 6.0;

        // 计算当前速度方向作为切向
        Point2f vel(x.at<double>(2), x.at<double>(3));
        Point2f tangential, normal;
        computeDirections(vel, tangential, normal);
        
        // 计算位置变化
        Point2f pos_delta = tangential * (x.at<double>(2)*dt + x.at<double>(4)*dt2 + x.at<double>(6)*dt3) +
                           normal * (x.at<double>(3)*dt + x.at<double>(5)*dt2 + x.at<double>(7)*dt3);
        
        result.at<double>(0) = x.at<double>(0) + pos_delta.x;  // 新位置x
        result.at<double>(1) = x.at<double>(1) + pos_delta.y;  // 新位置y
        
        // 速度、加速度和加加速度的更新
        result.at<double>(2) = x.at<double>(2) + x.at<double>(4)*dt + x.at<double>(6)*dt2;  // vt
        result.at<double>(3) = x.at<double>(3) + x.at<double>(5)*dt + x.at<double>(7)*dt2;  // vn
        result.at<double>(4) = x.at<double>(4) + x.at<double>(6)*dt;  // at
        result.at<double>(5) = x.at<double>(5) + x.at<double>(7)*dt;  // an
        result.at<double>(6) = x.at<double>(6);  // jt
        result.at<double>(7) = x.at<double>(7);  // jn

        return result;
    }

    Mat jacobianF() {
        Mat F = Mat::eye(8, 8, CV_64F);
        double dt2 = dt * dt / 2.0;
        double dt3 = dt * dt * dt / 6.0;
        
        // x方向导数
        F.at<double>(0,2) = dt;   // dx/dvx
        F.at<double>(0,4) = dt2;  // dx/dax
        F.at<double>(0,6) = dt3;  // dx/djx
        F.at<double>(2,4) = dt;   // dvx/dax
        F.at<double>(2,6) = dt2;  // dvx/djx
        F.at<double>(4,6) = dt;   // dax/djx
        
        // y方向导数
        F.at<double>(1,3) = dt;   // dy/dvy
        F.at<double>(1,5) = dt2;  // dy/day
        F.at<double>(1,7) = dt3;  // dy/djy
        F.at<double>(3,5) = dt;   // dvy/day
        F.at<double>(3,7) = dt2;  // dvy/djy
        F.at<double>(5,7) = dt;   // day/djy
        
        return F;
    }

    void predict() {
        // 预测状态
        state = f(state);
        // 预测协方差
        Mat F = jacobianF();
        P = F * P * F.t() + Q;
    }

    void update(const Point2f& measurement, int current_frame_id) {
        // 根据帧ID更新时间步长
        if(last_frame_id < 0) {
            dt = 1.0/30.0;  // 首帧使用默认值
        } else {
            dt = (current_frame_id - last_frame_id) * (1.0/30.0);
        }
        last_frame_id = current_frame_id;  // 保存当前帧ID作为下一帧的上一帧ID
        Mat z = (Mat_<double>(2,1) << measurement.x, measurement.y);
        Mat H = (Mat_<double>(2,8) << 1,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0);  // 测量矩阵
        
        Mat y = z - H * state;
        Mat S = H * P * H.t() + R;
        Mat K = P * H.t() * S.inv();
        
        state = state + K * y;
        // 确保时间步长非负
        if(dt < 0) dt = 1.0/30.0;
        P = (Mat::eye(8,8,CV_64F) - K * H) * P;
    }

    Point2f getPredictedPosition() {
        return Point2f(state.at<double>(0), state.at<double>(1));
    }

    Point2f getTangentialVelocity() {
        Point2f tangential, normal;
        computeDirections(Point2f(state.at<double>(2), state.at<double>(3)), tangential, normal);
        return tangential * state.at<double>(2);
    }

    Point2f getNormalVelocity() {
        Point2f tangential, normal;
        computeDirections(Point2f(state.at<double>(2), state.at<double>(3)), tangential, normal);
        return normal * state.at<double>(3);
    }

    Point2f getPredictedAcceleration() {
        return Point2f(state.at<double>(4), state.at<double>(5));
    }
};

class LightBarDetector {
private:
    DetectionParams params;
    Mat all_rect_debug;
    vector<Point2f> history_centers;  // 只保留历史中心点
    bool tracking_active = false;

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
            cout << "灯条中心距/平均长度比值: " << ratio << endl;
            return ratio;
        };
        
        float ratio1 = getRatio(p1.left, p1.right), ratio2 = getRatio(p2.left, p2.right);
        if(ratio1 < 1.5f || ratio1 > 3.5f || ratio2 < 1.5f || ratio2 > 3.5f) {
            cout << "装甲板宽高比不合适: ratio1=" << ratio1 << ", ratio2=" << ratio2 
                 << " (应在1.5~5.0之间)" << endl;
            return false;
        }
        
        auto getAngle = [](const Point2f& bottom, const Point2f& top) {
            float angle = atan2(-(top.y - bottom.y), top.x - bottom.x);
            // 将角度规范化到[-π/2, π/2]范围内
            while(angle > CV_PI/2) angle -= CV_PI;
            while(angle < -CV_PI/2) angle += CV_PI;
            float abs_angle = abs(angle);  // 取绝对值处理，因为我们只关心倾斜程度
            return abs_angle;
        };
        
        // 计算左右灯条的角度
        float angle_left = getAngle(p1.left.bottom_point, p1.left.top_point);
        float angle_right = getAngle(p1.right.bottom_point, p1.right.top_point);

        // 只检查两个灯条是否近似平行(允许15度误差)
        float parallel_diff = abs(angle_left - angle_right);
        if(parallel_diff > CV_PI/18) { // 15度
            cout << "左右灯条不平行, 角度差: " << parallel_diff * 180/CV_PI << "° (应小于15°)" << endl;
            cout << "左灯条角度: " << angle_left * 180/CV_PI << "°" << endl;
            cout << "右灯条角度: " << angle_right * 180/CV_PI << "°" << endl;
            return false;
        }

        float avg_angle = (angle_left + angle_right) * 0.5f;
        float angle_deg = avg_angle * 180.0f / CV_PI;
        
        cout << "左右灯条角度: left=" << angle_left*180/CV_PI << "°, right=" << angle_right*180/CV_PI 
             << "°, avg=" << angle_deg << "°" << endl;
        
        float heightDiffThreshold = 2.5f + (min(abs(angle_deg), 30.0f) / 30.0f) * 0.5f;
        float heightDiff1 = abs(p1.left.rect.center.y - p1.right.rect.center.y);
        float heightDiff2 = abs(p2.left.rect.center.y - p2.right.rect.center.y);
        float avgLen1 = (p1.left.length + p1.right.length) * 0.5f;
        float avgLen2 = (p2.left.length + p2.right.length) * 0.5f;
        
        if(heightDiff1 > heightDiffThreshold * avgLen1 || heightDiff2 > heightDiffThreshold * avgLen2) {
            cout << "高度差过大: diff1/avgLen1=" << heightDiff1/avgLen1 
                 << ", diff2/avgLen2=" << heightDiff2/avgLen2
                 << " (阈值: " << heightDiffThreshold << ")" << endl;
            return false;
        }
        
        float angle1 = atan2(p1.right.rect.center.y - p1.left.rect.center.y, 
                            p1.right.rect.center.x - p1.left.rect.center.x);
        float angle2 = atan2(p2.right.rect.center.y - p2.left.rect.center.y, 
                            p2.right.rect.center.x - p2.left.rect.center.x);
        
        if(abs(angle1 - angle2) > CV_PI/6) {
            cout << "装甲板倾角差过大: " << abs(angle1 - angle2)*180/CV_PI 
                 << "° (应小于30°)" << endl;
            return false;
        }
        
        Point2f center1 = p1.left.rect.center;
        Point2f center2 = p2.left.rect.center;
        float dist = norm(center1 - center2);
        if(dist >= threshold) {
            cout << "中心点距离过大: " << dist << " (阈值: " << threshold << ")" << endl;
            return false;
        }

        // 获取装甲板四个角点
        vector<Point2f> corners = {
            p1.left.top_point,      // 左上
            p1.right.top_point,     // 右上
            p1.right.bottom_point,  // 右下
            p1.left.bottom_point    // 左下
        };

        // 检查四边形临边夹角
        for(int i = 0; i < 4; i++) {
            Point2f v1 = corners[(i+1)%4] - corners[i];
            Point2f v2 = corners[(i+3)%4] - corners[i];
            float angle = acos(v1.dot(v2) / (norm(v1) * norm(v2)));
            float angle_deg = angle * 180.0f / CV_PI;
            
            if(angle_deg < 45.0f) {
                cout << "装甲板四边形临边夹角过小: " << angle_deg 
                     << "° (应大于45°)" << endl;
                return false;
            }
        }
        
        cout << "匹配成功!" << endl;
        return true;
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
        
        if(pairs.empty()) {
            imshow("Debug Window", debug_img);
            return;
        }

        // 获取所有装甲板中心点
        vector<Point2f> centers;
        for(const auto& p : pairs) {
            centers.push_back((p.left.rect.center + p.right.rect.center) * 0.5f);
        }
        
        // 使用MotionEstimator处理点
        MotionEstimator estimator;
        estimator.processFrame(frame);
        Mat classified_points = estimator.getClassifiedPoints();
        
        // 绘制当前检测到的点(红色)
        for(const auto& center : centers) {
            circle(debug_img, center, 3, Scalar(0,0,255), -1);
            
            // 在中心点上方显示预测结果
        Point2f pred_point(classified_points.at<float>(0,0), 
                          classified_points.at<float>(0,1));
        putText(debug_img, "Pred: (" + to_string((int)pred_point.x) + "," + to_string((int)pred_point.y) + ")", 
                Point(center.x + 10, center.y - 10), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0), 2);
        }
        
        // 绘制预测点(黄色)
        for (int i = 0; i < classified_points.rows; i++) {
            Point2f pred_point(classified_points.at<float>(i,0), 
                              classified_points.at<float>(i,1));
            circle(debug_img, pred_point, 4, Scalar(0,255,255), -1);
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
                    Rect bbox = r.boundingRect();
        if(bbox.x >= 0 && bbox.y >= 0 && 
           bbox.x + bbox.width <= frame.cols && 
           bbox.y + bbox.height <= frame.rows) {
            lights.emplace_back(r, mean(gray(bbox))[0], mean(frame(bbox)));
        } else {
            cerr << "无效的边界框: " << bbox << endl;
        }
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
    VideoCapture cap("2.mp4");
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
        
        char key = waitKey(10);  // 降低延迟，提高显示帧率
        if(key == 27) break;
        else if(key == 's') {
            imwrite("light_bars.jpg", detector.debug_img);
        }
    }
    
    destroyAllWindows();
    return 0;
}