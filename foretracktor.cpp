#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace cv;
using namespace std;

struct DetectionParams {
    int gaussian_size = 3;
    int threshold_value = 100;
    int min_contour_area = 50;
    int brightness_threshold = 200; 
    int bilateral_d = 7;
    double bilateral_sigma_color = 75;
    double bilateral_sigma_space = 75;
    int morph_open_size = 0;
    int dilate_iterations = 0;
    float min_aspect_ratio = 2.5f;   // 修改最小长宽比
    float max_aspect_ratio = 10.0f;  // 保持最大长宽比
};

class ExtendedKalmanFilter {
private:
    Mat state;         // 状态向量 [x, y, z, vt, vn, vz, at, an, az, jt, jn, jz]
    Mat P;             // 状态协方差矩阵
    Mat Q;             // 过程噪声协方差
    Mat R;             // 测量噪声协方差
    double dt;         // 时间步长
    double last_timestamp;  // 记录上一帧时间戳（秒）
    Point3f last_tangential;  // 上一次的切向量
    Point3f last_normal;     // 上一次的法向量
    Point3f last_binormal;   // 上一次的副法向量

    // 计算切向、法向和副法向单位向量
    void computeDirections(const Point3f& vel, Point3f& tangential, Point3f& normal, Point3f& binormal) {
        float vel_norm = norm(vel);
        if(vel_norm < 1e-6) {
            // 如果速度太小,使用上一次的方向
            tangential = last_tangential;
            normal = last_normal;
            binormal = last_binormal;
            return;
        }
        tangential = vel / vel_norm;
        
        // 计算法向量
        if(abs(tangential.x) < 0.9) {
            normal = Point3f(-tangential.y, tangential.x, 0);
        } else {
            normal = Point3f(0, -tangential.z, tangential.y);
        }
        normal = normal / norm(normal);
        
        // 计算副法向量
        binormal = tangential.cross(normal);
        
        last_tangential = tangential;
        last_normal = normal;
        last_binormal = binormal;
    }

public:
    ExtendedKalmanFilter() {
        state = Mat::zeros(12, 1, CV_64F);
        P = Mat::eye(12, 12, CV_64F) * 100;
        // 调整过程噪声：切向速度噪声较小,法向加速度噪声较大
        Q = Mat::eye(12, 12, CV_64F);
        Q.at<double>(3,3) *= 0.1;  // 切向速度
        Q.at<double>(4,4) *= 0.5;  // 法向速度
        Q.at<double>(5,5) *= 0.3;  // z方向速度
        Q.at<double>(6,6) *= 0.2;  // 切向加速度
        Q.at<double>(7,7) *= 1.0;  // 法向加速度
        Q.at<double>(8,8) *= 0.5;  // z方向加速度
        R = Mat::eye(3, 3, CV_64F) * 1.0;
        dt = 1.0/30.0;
        last_timestamp = -1;  // 初始化为无效值
        last_tangential = Point3f(1, 0, 0);
        last_normal = Point3f(0, 1, 0);
        last_binormal = Point3f(0, 0, 1);
    }

    void init(const Point3f& pos, const Point3f& vel = Point3f(0, 0, 0)) {
        Point3f tangential, normal, binormal;
        computeDirections(vel, tangential, normal, binormal);
        
        float vel_mag = norm(vel);
        float vt = vel_mag * tangential.dot(vel/vel_mag);
        float vn = vel_mag * normal.dot(vel/vel_mag);
        float vz = vel_mag * binormal.dot(vel/vel_mag);

        state.at<double>(0) = pos.x;        // x位置
        state.at<double>(1) = pos.y;        // y位置
        state.at<double>(2) = pos.z;        // z位置
        state.at<double>(3) = vt;           // 切向速度
        state.at<double>(4) = vn;           // 法向速度
        state.at<double>(5) = vz;           // z方向速度
        state.at<double>(6) = 0;            // 切向加速度
        state.at<double>(7) = 0;            // 法向加速度
        state.at<double>(8) = 0;            // z方向加速度
        state.at<double>(9) = 0;            // 切向加加速度
        state.at<double>(10) = 0;           // 法向加加速度
        state.at<double>(11) = 0;           // z方向加加速度
    }
    
    void init(const Point2f& pos, const Point2f& vel = Point2f(0, 0)) {
        Point3f vel3f(vel.x, vel.y, 0);
        Point3f pos3f(pos.x, pos.y, 0);
        init(pos3f, vel3f);
    }

    Mat f(const Mat& x) {
        Mat result = Mat::zeros(12, 1, CV_64F);
        double dt2 = dt * dt / 2.0;
        double dt3 = dt * dt * dt / 6.0;

        // 计算当前速度方向作为切向
        Point3f vel(x.at<double>(3), x.at<double>(4), x.at<double>(5));
        Point3f tangential, normal, binormal;
        computeDirections(vel, tangential, normal, binormal);
        
        // 计算位置变化
        Point3f pos_delta = tangential * (x.at<double>(3)*dt + x.at<double>(6)*dt2 + x.at<double>(9)*dt3) +
                           normal * (x.at<double>(4)*dt + x.at<double>(7)*dt2 + x.at<double>(10)*dt3) +
                           binormal * (x.at<double>(5)*dt + x.at<double>(8)*dt2 + x.at<double>(11)*dt3);
        
        result.at<double>(0) = x.at<double>(0) + pos_delta.x;  // 新位置x
        result.at<double>(1) = x.at<double>(1) + pos_delta.y;  // 新位置y
        result.at<double>(2) = x.at<double>(2) + pos_delta.z;  // 新位置z
        
        // 速度、加速度和加加速度的更新
        result.at<double>(3) = x.at<double>(3) + x.at<double>(6)*dt + x.at<double>(9)*dt2;  // vt
        result.at<double>(4) = x.at<double>(4) + x.at<double>(7)*dt + x.at<double>(10)*dt2;  // vn
        result.at<double>(5) = x.at<double>(5) + x.at<double>(8)*dt + x.at<double>(11)*dt2;  // vz
        result.at<double>(6) = x.at<double>(6) + x.at<double>(9)*dt;  // at
        result.at<double>(7) = x.at<double>(7) + x.at<double>(10)*dt;  // an
        result.at<double>(8) = x.at<double>(8) + x.at<double>(11)*dt;  // az
        result.at<double>(9) = x.at<double>(9);  // jt
        result.at<double>(10) = x.at<double>(10);  // jn
        result.at<double>(11) = x.at<double>(11);  // jz

        return result;
    }

    Mat jacobianF() {
        Mat F = Mat::eye(12, 12, CV_64F);
        double dt2 = dt * dt / 2.0;
        double dt3 = dt * dt * dt / 6.0;
        
        // x方向导数
        F.at<double>(0,3) = dt;   // dx/dvx
        F.at<double>(0,6) = dt2;  // dx/dax
        F.at<double>(0,9) = dt3;  // dx/djx
        F.at<double>(3,6) = dt;   // dvx/dax
        F.at<double>(3,9) = dt2;  // dvx/djx
        F.at<double>(6,9) = dt;   // dax/djx
        
        // y方向导数
        F.at<double>(1,4) = dt;   // dy/dvy
        F.at<double>(1,7) = dt2;  // dy/day
        F.at<double>(1,10) = dt3;  // dy/djy
        F.at<double>(4,7) = dt;   // dvy/day
        F.at<double>(4,10) = dt2;  // dvy/djy
        F.at<double>(7,10) = dt;   // day/djy
        
        // z方向导数
        F.at<double>(2,5) = dt;   // dz/dvz
        F.at<double>(2,8) = dt2;  // dz/daz
        F.at<double>(2,11) = dt3;  // dz/djz
        F.at<double>(5,8) = dt;   // dvz/daz
        F.at<double>(5,11) = dt2;  // dvz/djz
        F.at<double>(8,11) = dt;   // daz/djz
        
        return F;
    }

    void predict() {
        // 预测状态
        state = f(state);
        // 预测协方差
        Mat F = jacobianF();
        P = F * P * F.t() + Q;
    }

    void update(const Point3f& measurement, double current_timestamp) {
        // 根据时间戳更新时间步长
        if(last_timestamp < 0) {
            dt = 1.0/30.0;  // 首帧使用默认值
        } else {
            dt = current_timestamp - last_timestamp;
        }
        last_timestamp = current_timestamp;  // 保存当前时间戳作为下一帧的上一帧时间戳
        Mat z = (Mat_<double>(3,1) << measurement.x, measurement.y, measurement.z);
        Mat H = (Mat_<double>(3,12) << 1,0,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0,0);  // 测量矩阵
        
        Mat y = z - H * state;
        Mat S = H * P * H.t() + R;
        Mat K = P * H.t() * S.inv();
        
        state = state + K * y;
        // 确保时间步长非负
        if(dt < 0) dt = 1.0/30.0;
        P = (Mat::eye(12,12,CV_64F) - K * H) * P;
    }

    Point3f getPredictedPosition() {
        return Point3f(state.at<double>(0), state.at<double>(1), state.at<double>(2));
    }

    Point3f getTangentialVelocity() {
        Point3f tangential, normal, binormal;
        computeDirections(Point3f(state.at<double>(3), state.at<double>(4), state.at<double>(5)), tangential, normal, binormal);
        return tangential * state.at<double>(3);
    }

    Point3f getNormalVelocity() {
        Point3f tangential, normal, binormal;
        computeDirections(Point3f(state.at<double>(3), state.at<double>(4), state.at<double>(5)), tangential, normal, binormal);
        return normal * state.at<double>(4);
    }

    Point3f getBinormalVelocity() {
        Point3f tangential, normal, binormal;
        computeDirections(Point3f(state.at<double>(3), state.at<double>(4), state.at<double>(5)), tangential, normal, binormal);
        return binormal * state.at<double>(5);
    }

    Point3f getPredictedAcceleration() {
        return Point3f(state.at<double>(6), state.at<double>(7), state.at<double>(8));
    }
    

    

};

class LightBarDetector {
private:
    DetectionParams params;
    Mat all_rect_debug;
    vector<Point2f> history_centers;  // 只保留历史中心点
    ExtendedKalmanFilter ekf;
    bool tracking_active = false;
    const int max_history = 5;  // 使用5帧历史数据
    const int predict_frames = 1;  // 只预测未来1帧
    
    // 初始化相机参数
    void initCameraParams() {
        // 相机内参矩阵 (根据用户提供的精确参数)
        camera_matrix = (Mat_<double>(3,3) << 
            1037.06391, 0, 306.61321,
            0, 1036.44396, 228.42354,
            0, 0, 1);
            
        // 畸变系数 (根据用户提供的精确参数)
        dist_coeffs = (Mat_<double>(1,5) << -0.108574, 0.260086, 0.000301, 0.001182, 0.000000);
        
        // 初始化去畸变映射
        initUndistortRectifyMap(camera_matrix, dist_coeffs, Mat(),
                              camera_matrix, Size(640, 480), CV_16SC2, map1, map2);
    }
    
public:
    LightBarDetector() {
        initCameraParams();  // 构造函数中初始化相机参数
    }
    

    
    // 颜色筛选参数
    int h_min = 91, h_max = 141;
    int s_min = 193, s_max = 255;
    int v_min = 97, v_max = 255;
    Mat color_mask;
    Mat hsv;
    
    // 相机标定参数
    Mat camera_matrix;
    Mat dist_coeffs;
    Mat map1, map2;
    
    // 装甲板3D模型点(4:1长宽比，10cm宽度)
    vector<Point3f> armor_3d_points = {
        Point3f(-0.2f, -0.05f, 0),  // 左上 (长20cm,宽5cm,符合4:1比例)
        Point3f(0.2f, -0.05f, 0),   // 右上
        Point3f(0.2f, 0.05f, 0),    // 右下
        Point3f(-0.2f, 0.05f, 0)    // 左下
    };
    
    // HSV滑动条回调函数
    static void onTrackbar(int, void* userdata) {
        LightBarDetector* detector = static_cast<LightBarDetector*>(userdata);
        detector->updateColorMask();
    }
    
    // 更新颜色掩膜
    void updateColorMask() {
        inRange(hsv, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), color_mask);
    }

    struct LightBar {
        RotatedRect rect;
        Point2f top_point, bottom_point;
        float length, brightness;
        Vec3b color;
        
        // 检测灯带边缘周围颜色
        bool isBlueHSV(const Mat& frame, const Rect& bbox) {
            // 计算膨胀20%后的区域
        float scale = 1.5f;
        int new_width = bbox.width * scale;
        int new_height = bbox.height * scale;
        int new_x = bbox.x - (new_width - bbox.width)/2;
        int new_y = bbox.y - (new_height - bbox.height)/2;
        
        // 创建膨胀后的矩形
        Rect expanded_rect(new_x, new_y, new_width, new_height);
        expanded_rect &= Rect(0, 0, frame.cols, frame.rows);
        
        // 计算边缘区域(膨胀区域减去原始区域)
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);
        rectangle(mask, expanded_rect, Scalar(255), FILLED);
        rectangle(mask, bbox, Scalar(0), FILLED);
        
        // 获取边缘区域轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // 将轮廓区域转换为矩形集合
        vector<Rect> regions;
        for(const auto& contour : contours) {
            regions.push_back(boundingRect(contour));
        }
            
            // 蓝色HSV范围参数
            Scalar lower_blue = Scalar(91, 193, 97);
            Scalar upper_blue = Scalar(141, 255, 255);
            
            Mat hsv, blue_mask;
            double total_pixels = 0;
            double blue_pixels = 0;
            
            for (const auto& region : regions) {
                if (region.area() > 0) {
                    Mat roi = frame(region);
                    cvtColor(roi, hsv, COLOR_BGR2HSV);
                    inRange(hsv, lower_blue, upper_blue, blue_mask);
                    blue_pixels += countNonZero(blue_mask);
                    total_pixels += region.area();
                }
            }
            
            if (total_pixels == 0) return false;
            
            double blue_ratio = blue_pixels / total_pixels;
            return blue_ratio > 0.05; 
        }
        
        
        bool isBlue() const {
            Mat hsv;
            Mat bgr(1, 1, CV_8UC3, color);
            cvtColor(bgr, hsv, COLOR_BGR2HSV);
            int h = hsv.at<Vec3b>(0,0)[0];
            int s = hsv.at<Vec3b>(0,0)[1];
            int v = hsv.at<Vec3b>(0,0)[2];
            
        return (h >= 91 && h <= 141) && (s >= 193) && (v >= 97);
        }

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
        // 计算两个灯条的中心距离与平均长度的比值
        auto getRatio = [](const LightBar& l1, const LightBar& l2) {
            float center_dist = norm(l1.rect.center - l2.rect.center);
            float avg_length = (l1.length + l2.length) * 0.5f;
            return center_dist / avg_length;
        };
        
        float ratio1 = getRatio(p1.left, p1.right), ratio2 = getRatio(p2.left, p2.right);
        if(ratio1 < 1.5f || ratio1 > 8.0f || ratio2 < 1.5f || ratio2 > 8.0f) {
            cout << "装甲板宽高比不合适: ratio1=" << ratio1 << ", ratio2=" << ratio2 
                 << " (应在1.5~5.0之间)" << endl;
            return false;
        }
        
        // 计算灯条的角度(弧度制)
        auto getAngle = [](const Point2f& bottom, const Point2f& top) {
            return atan2(-(top.y - bottom.y), top.x - bottom.x);
        };
        

        // 检查灯条平行度和高度差
        auto checkParallelAndHeight = [getAngle](const ArmorPair& p) {
            float angle_left = getAngle(p.left.bottom_point, p.left.top_point);
            float angle_right = getAngle(p.right.bottom_point, p.right.top_point);
            
            // 归一化角度到[-π/2, π/2]
            angle_left = fmod(angle_left + CV_PI/2, CV_PI) - CV_PI/2;
            angle_right = fmod(angle_right + CV_PI/2, CV_PI) - CV_PI/2;
            
            float parallel_diff = abs(angle_left - angle_right);
            if(parallel_diff > CV_PI/12) {
                return false;
            }
            
            float avg_angle = (angle_left + angle_right) * 0.5f;
            float angle_deg = abs(avg_angle * 180.0f / CV_PI);
            
            float heightDiffThreshold = 2.5f + (min(angle_deg, 30.0f) / 30.0f) * 0.5f;
            float heightDiff = abs(p.left.rect.center.y - p.right.rect.center.y);
            float avgLen = (p.left.length + p.right.length) * 0.5f;
            
            return heightDiff <= heightDiffThreshold * avgLen;
        };
        
        if(!checkParallelAndHeight(p1) || !checkParallelAndHeight(p2)) {
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
        float dist = norm(center1 - center2);
        if(dist >= threshold) {

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

                return false;
            }
        }
        

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

    

    // 预测未来几帧的位置
    vector<Point2f> predictFuturePositions(const Point2f& current_pos, const Point2f& velocity) {
        vector<Point2f> future_positions;
        for(int i = 1; i <= predict_frames; i++) {
            future_positions.push_back(current_pos + velocity * i);
        }
        return future_positions;
    }

public:
    Mat debug_img;
    
    
    Mat undistortImage(const Mat& frame) {
        Mat undistorted;
        remap(frame, undistorted, map1, map2, INTER_LINEAR);
        return undistorted;
    }
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
        static int lost_frames = 0; // 连续丢失帧数计数器
        debug_img = frame.clone();
        auto pairs = matchLightBars(lights);
        
        // 绘制所有检测到的灯带
        for(const auto& light : lights) {
            Point2f vertices[4];
            light.rect.points(vertices);
            for(int i = 0; i < 4; i++) {
                line(debug_img, vertices[i], vertices[(i+1)%4], Scalar(0,0,255), 2);
            }
        }
        
        if(pairs.empty()) {
            lost_frames++; // 增加丢失帧数
            
            // 如果连续丢失超过2帧，则停止追踪
            if(lost_frames >= 2 && tracking_active) {
                tracking_active = false;
                history_centers.clear();
                if(++lost_frames >= 2) {
                    tracking_active = false;
                    lost_frames = 0;
                }
                imshow("Debug Window", debug_img);
                return;
            }
            
            if(tracking_active && !history_centers.empty()) {
                // 使用历史数据进行预测
                Point2f last_center = history_centers.back();
                Point2f velocity = history_centers.size() >= 2 ? 
                    (last_center - history_centers[history_centers.size()-2]) : Point2f(0,0);
                
                vector<Point2f> future_positions = predictFuturePositions(last_center, velocity);
                
                // 绘制预测轨迹
                Point2f last_pos = last_center;
                for(size_t i = 0; i < future_positions.size(); i++) {
                    int alpha = 255 * (predict_frames - i) / predict_frames;
                    Scalar color(0, alpha, 0);
                    circle(debug_img, future_positions[i], 4, color, -1);
                    line(debug_img, last_pos, future_positions[i], color, 1, LINE_AA);
                    last_pos = future_positions[i];
                }
                
                // 仅显示预测点坐标
                putText(debug_img, 
                        format("(%.1f, %.1f)", future_positions[0].x, future_positions[0].y),
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
            }
            imshow("Debug Window", debug_img);
            return;
        }
        
        // 检测到装甲板时重置丢失帧数
        lost_frames = 0;

        const auto& p = pairs[0];
        Point2f center = (p.left.rect.center + p.right.rect.center) * 0.5f;
        circle(debug_img, center, 3, Scalar(0,0,255), -1);  // 当前点
        
        // 获取装甲板4个图像角点
        // 确保装甲板角点顺序与3D模型点一致
        vector<Point2f> image_points = {
            p.left.top_point,      // 左上 [-0.2, 0.05, 0]
            p.right.top_point,     // 右上 [0.2, 0.05, 0]
            p.right.bottom_point,  // 右下 [0.2, -0.05, 0]
            p.left.bottom_point    // 左下 [-0.2, -0.05, 0]
        };
        
        // 验证并确保角点顺序与3D模型严格对应
        vector<Point2f> sorted_points(4);
        
        // 找到左上角点(最小x+y)
        int top_left_idx = 0;
        for(int i = 1; i < 4; i++) {
            if(image_points[i].x + image_points[i].y < image_points[top_left_idx].x + image_points[top_left_idx].y) {
                top_left_idx = i;
            }
        }
        sorted_points[0] = image_points[top_left_idx];
        
        // 找到与左上角点距离最近的两个点作为右上和左下
        vector<pair<float,int>> distances;
        for(int i = 0; i < 4; i++) {
            if(i != top_left_idx) {
                distances.emplace_back(norm(image_points[i] - sorted_points[0]), i);
            }
        }
        sort(distances.begin(), distances.end());
        
        // 确定右上和左下角点
        Point2f vec1 = image_points[distances[0].second] - sorted_points[0];
        Point2f vec2 = image_points[distances[1].second] - sorted_points[0];
        float cross = vec1.x * vec2.y - vec1.y * vec2.x;
        
        if(cross > 0) {
            sorted_points[1] = image_points[distances[0].second];
            sorted_points[3] = image_points[distances[1].second];
        } else {
            sorted_points[1] = image_points[distances[1].second];
            sorted_points[3] = image_points[distances[0].second];
        }
        
        // 最后一个点是右下角点
        for(int i = 0; i < 4; i++) {
            if(i != top_left_idx && i != distances[0].second && i != distances[1].second) {
                sorted_points[2] = image_points[i];
                break;
            }
        }
        
        image_points = sorted_points;
        
        
        // 使用solvePnP计算位姿
        Mat rvec, tvec;
        if(armor_3d_points.size() != 4 || image_points.size() != 4) {
            cerr << "错误: objectPoints或imagePoints数量不正确" << endl;
            return;
        }
        if(!camera_matrix.data || !dist_coeffs.data) {
            cerr << "错误: 相机参数未初始化" << endl;
            return;
        }
        solvePnP(armor_3d_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
        
       
        
        // 计算距离(转换为厘米)
        double distance = norm(tvec) * 100;  // 转换为厘米，tvec单位是米
        
        // 显示距离和位姿信息
        string distance_text = format("Distance: %.1f cm", distance);
        string pose_text = format("Pose: [%.1f, %.1f, %.1f]", 
                                 tvec.at<double>(0)*0.1, tvec.at<double>(1)*0.1, tvec.at<double>(2)*0.1);
        
        putText(debug_img, distance_text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
        putText(debug_img, pose_text, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);

        if(!tracking_active) {
            ekf.init(center);
            tracking_active = true;
            history_centers.clear();
        }

        history_centers.push_back(center);
        if(history_centers.size() > 3) {  // 只保留最近3帧
            history_centers.erase(history_centers.begin());
        }

        // 绘制历史轨迹（最近3帧）
        for(size_t i = 1; i < history_centers.size(); i++) {
            line(debug_img, history_centers[i-1], history_centers[i], 
                 Scalar(255,0,0), 2);  // 蓝色历史轨迹
        }

        // 预测未来3帧
        if(tracking_active && history_centers.size() >= 2) {
            Point2f velocity = (center - history_centers[history_centers.size()-2]);
            vector<Point2f> future_positions = predictFuturePositions(center, velocity);
            
            // 绘制预测轨迹
            Point2f last_pos = center;
            for(size_t i = 0; i < future_positions.size(); i++) {
                // 使用渐变色显示未来轨迹
                int alpha = 255 * (predict_frames - i) / predict_frames;
                Scalar color(0, alpha, 0);
                
                // 绘制预测点
                circle(debug_img, future_positions[i], 4, color, -1);
                
                // 绘制预测轨迹线
                line(debug_img, last_pos, future_positions[i], color, 1, LINE_AA);
                last_pos = future_positions[i];
            }

            // 特别标记下一帧位置
            circle(debug_img, future_positions[0], 1, Scalar(0,255,0), -1);
            
            // 仅显示预测点坐标
            putText(debug_img, 
                    format("(%.1f, %.1f)", future_positions[0].x, future_positions[0].y),
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
        }

        
        
        imshow("Debug Window", debug_img);
    }

    vector<LightBar> detect(Mat& frame) {
        auto lights = vector<LightBar>();
        all_rect_debug = frame.clone();  // 初始化调试图像
        debug_img = frame.clone();  // 移动到函数开始处，用于绘制所有矩形
        
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        

        
        updateColorMask();

        
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 预处理流程: 亮度筛选->二值化->面积筛选
        Mat brightMask = getBrightnessFilteredImage(gray);
        
        Mat bin;
        threshold(gray, bin, params.threshold_value, 255, THRESH_BINARY);
        bin = bin & brightMask;
        
        Mat areaMask = getAreaFilteredImage(bin);
        bin = bin & areaMask;
        
       // 查找轮廓
        vector<vector<Point>> contours;
        findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        
        
        for(const auto& c : contours) {
            try {
                RotatedRect r = minAreaRect(c);
                Point2f vertices[4];
                r.points(vertices);

                // 计算最小外接矩形的长宽比
                float long_side = max(r.size.width, r.size.height);
                float short_side = min(r.size.width, r.size.height);
                float aspect_ratio = long_side / short_side;
                
                // 只要长宽比符合要求,就认为是灯带
                if(aspect_ratio >= params.min_aspect_ratio && aspect_ratio <= params.max_aspect_ratio) {
                    Rect bbox = r.boundingRect();
                    if(bbox.x >= 0 && bbox.y >= 0 && 
                       bbox.x + bbox.width <= frame.cols && 
                       bbox.y + bbox.height <= frame.rows) {
                        LightBar bar(r, mean(gray(bbox))[0], mean(frame(bbox)));
                        if(bar.isBlueHSV(frame, bbox)) {  // 使用HSV颜色空间判断蓝色灯带
                            // 绘制找到的灯带
                            for(int i = 0; i < 4; i++) {
                                line(all_rect_debug, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 2);
                            }
                            lights.push_back(bar);
                        }
                    }
            
                }
            } catch(...) {
                cout << "处理轮廓时出现异常" << endl;
            }
        }
        

        showDebugInfo(frame, lights);

        return lights;
    }
};

int main() {
    VideoCapture cap("lights.mp4");
    if(!cap.isOpened()) {
        cout << "无法打开视频文件！" << endl;
        return -1;
    }

    LightBarDetector detector;
    Mat frame;
    
    namedWindow("Debug Window", WINDOW_AUTOSIZE);  
    
    
    while(true) {
        if(!cap.read(frame)) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        detector.detect(frame);
        
        char key = waitKey(10);  
        if(key == 27) break;
        else if(key == 's') {
            imwrite("light_bars.jpg", detector.debug_img);
        }
    }
    
    destroyAllWindows();
    return 0;
}