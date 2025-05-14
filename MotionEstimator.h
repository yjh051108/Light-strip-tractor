#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

class MotionEstimator {
private:
    /**
     * @brief 跟踪点数据结构
     */
    struct TrackedPoint {
        cv::Point2f position;      // 当前位置
        cv::Point2f velocity;      // 当前速度
        cv::Point2f acceleration;  // 当前加速度
        std::vector<cv::Point2f> history; // 历史位置记录
        int life_time = 0;         // 生命周期计数器
    };
    
    std::map<int, TrackedPoint> tracked_points; // 跟踪点映射表
    int next_id = 0;               // 下一个跟踪点ID
    
    // 视觉里程计相关
    cv::Mat prev_frame;            // 前一帧图像
    cv::Mat curr_frame;            // 当前帧图像
    cv::Mat next_frame;            // 下一帧图像
    
    // 扩展卡尔曼滤波器
    cv::KalmanFilter kf;           // 卡尔曼滤波器实例
    
    /**
     * @brief 初始化卡尔曼滤波器参数
     */
    void initKalmanFilter();
    
    /**
     * @brief 使用3帧视觉里程计计算初始运动
     * @param points 特征点集合
     */
    void calculateInitialMotion(const std::vector<cv::Point2f>& points);
    
    /**
     * @brief 使用泰勒级数拟合预测运动
     * @param id 跟踪点ID
     * @return 预测的下一个位置
     */
    cv::Point2f predictWithTaylorSeries(int id);
    
    /**
     * @brief 使用扩展卡尔曼滤波更新状态
     * @param id 跟踪点ID
     * @param observed 观测到的位置
     */
    void updateWithEKF(int id, const cv::Point2f& observed);
    
public:
    /**
     * @brief 构造函数
     */
    MotionEstimator();
    
    /**
     * @brief 处理新帧
     * @param frame 输入图像帧
     */
    void processFrame(const cv::Mat& frame);
    
    /**
     * @brief 获取分类后的点矩阵
     * @return 分类后的点矩阵
     */
    cv::Mat getClassifiedPoints() const;
};