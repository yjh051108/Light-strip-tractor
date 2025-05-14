#include "MotionEstimator.h"
#include <opencv2/video/tracking.hpp>
#include <iostream>

MotionEstimator::MotionEstimator() {
    initKalmanFilter();
}

void MotionEstimator::initKalmanFilter() {
    // 4状态变量(x,y,vx,vy), 2观测变量(x,y)
    kf = cv::KalmanFilter(4, 2, 0);
    
    // 转移矩阵 (假设匀速运动模型)
    kf.transitionMatrix = (cv::Mat_<float>(4,4) << 
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1);
    
    // 初始化其他卡尔曼参数
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

void MotionEstimator::calculateInitialMotion(const std::vector<cv::Point2f>& points) {
    if (prev_frame.empty() || curr_frame.empty() || next_frame.empty()) return;
    
    // 计算光流
    std::vector<cv::Point2f> prev_points, next_points;
    std::vector<uchar> status;
    std::vector<float> err;
    
    cv::calcOpticalFlowPyrLK(prev_frame, curr_frame, points, next_points, status, err);
    
    // 计算速度和加速度
    for (size_t i = 0; i < points.size(); ++i) {
        if (status[i]) {
            cv::Point2f velocity = next_points[i] - points[i];
            tracked_points[next_id].velocity = velocity;
            tracked_points[next_id].acceleration = velocity / (1.0/30.0); // 假设帧率为30fps
            next_id++;
        }
    }
}

cv::Point2f MotionEstimator::predictWithTaylorSeries(int id) {
    auto& point = tracked_points[id];
    
    // 三阶泰勒级数展开预测
    float dt = 1.0/30.0; // 假设帧率为30fps
    float dt2 = dt * dt / 2.0f;
    float dt3 = dt * dt * dt / 6.0f;
    
    cv::Point2f predicted = point.position + 
                           point.velocity * dt + 
                           point.acceleration * dt2;
    
    return predicted;
}

void MotionEstimator::updateWithEKF(int id, const cv::Point2f& observed) {
    // 扩展卡尔曼滤波更新
    cv::Mat measurement = (cv::Mat_<float>(2,1) << observed.x, observed.y);
    
    // 预测
    cv::Mat prediction = kf.predict();
    
    // 更新过程噪声协方差矩阵
    kf.processNoiseCov = (cv::Mat_<float>(4,4) << 
        1e-4, 0, 0, 0,
        0, 1e-4, 0, 0,
        0, 0, 1e-2, 0,
        0, 0, 0, 1e-2);
    
    // 更新测量噪声协方差矩阵
    kf.measurementNoiseCov = (cv::Mat_<float>(2,2) << 
        1e-1, 0,
        0, 1e-1);
    
    // 更新
    cv::Mat estimated = kf.correct(measurement);
    
    // 更新跟踪点状态
    auto& point = tracked_points[id];
    point.position = cv::Point2f(estimated.at<float>(0), estimated.at<float>(1));
    point.velocity = cv::Point2f(estimated.at<float>(2), estimated.at<float>(3));
    point.history.push_back(point.position);
}

void MotionEstimator::processFrame(const cv::Mat& frame) {
    // 更新帧序列
    prev_frame = curr_frame;
    curr_frame = next_frame;
    next_frame = frame.clone();
    
    // 检测特征点 (这里简化处理)
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(frame, points, 100, 0.01, 10);
    
    // 计算初始运动
    calculateInitialMotion(points);
    
    // 对每个点进行预测和更新
    for (auto& p : points) {
        // 查找最近跟踪点或创建新跟踪点
        // 这里可以添加数据关联逻辑
        
        // 预测
        cv::Point2f predicted = predictWithTaylorSeries(0); // 简化处理
        
        // 更新
        updateWithEKF(0, p); // 简化处理
    }
}

cv::Mat MotionEstimator::getClassifiedPoints() const {
    // 将分类后的点组织成矩阵
    cv::Mat result(tracked_points.size(), 2, CV_32F);
    
    int i = 0;
    for (auto& pair : tracked_points) {
        result.at<float>(i, 0) = pair.second.position.x;
        result.at<float>(i, 1) = pair.second.position.y;
        i++;
    }
    
    return result;
}