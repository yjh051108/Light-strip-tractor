#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class MotionPredictor {
private:
    cv::Mat states;    // 多目标状态矩阵 [N x 8], 每行代表一个目标的状态向量 [x, y, vx, vy, ax, ay, jx, jy]
    cv::Mat Ps;        // 多目标状态协方差矩阵 [N x 8 x 8]
    cv::Mat Q;         // 过程噪声协方差 [8 x 8]
    cv::Mat R;         // 测量噪声协方差 [2 x 2]
    double dt;         // 时间步长
    cv::Mat history_points; // 多目标历史点矩阵 [N x M x 2], M为历史帧数
    cv::Mat predicted_points; // 预测点矩阵
    cv::Mat points;     // 当前点矩阵
    
    cv::Mat f(const cv::Mat& x);
    cv::Mat jacobianF();
    
public:
    MotionPredictor();
    
    // 初始化滤波器
    void init(const cv::Point2f& pos);
    
    // 更新历史数据点(最多保留6个)
    void updateHistory(const cv::Point2f& point);
    
    // 执行预测
    cv::Mat predictNextPoints(); // 返回预测点矩阵 [N x 2]
    
    // 预测单个点
    cv::Point2f predictNextPoint() {
        cv::Mat predicted = predictNextPoints();
        return cv::Point2f(predicted.at<double>(0,0), predicted.at<double>(0,1));
    }
    
    // 更新测量值
    void update(const cv::Point2f& measurement);
    
    // 执行预测
    void predict();
    
    // 获取历史数据点数量
    size_t getHistorySize() const;
};