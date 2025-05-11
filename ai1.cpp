#include <opencv2/opencv.hpp> 
#include <iostream>
#include <vector>
#include <omp.h>
using namespace cv;
using namespace std;

// 参数结构体
struct DetectionParams {
    int gaussian_size = 3;
    int threshold_value = 100;
    int min_contour_area = 50;
    int brightness_threshold = 58;
    int bilateral_d = 7;
    double bilateral_sigma_color = 75;
    double bilateral_sigma_space = 75;
    int morph_open_size = 3;
    int dilate_iterations = 2;
};

// 修改ArmorDetector类，移除运动轨迹相关成员
class LightBarDetector {
private:
    DetectionParams params;
    Mat debug_img;
    
    // 静态常量
    static constexpr int HISTORY_FRAME_COUNT = 180;
    static constexpr float UNBIND_THRESHOLD = 0.95f;
    static constexpr float MOTION_DEVIATION_THRESHOLD = 0.5f;
    static constexpr float MOTION_SIMILARITY_THRESHOLD = 0.998f;
    static constexpr float MAX_MATCH_DISTANCE = 50.0f;  // 最大匹配距离
    static constexpr float MAX_ANGLE_DIFF = 15.0f;      // 最大角度差异(度)
    static constexpr float MIN_TRACK_QUALITY = 0.6f;    // 最小跟踪质量
    static constexpr float MIN_PAIR_SCORE = 0.5f;       // 降低最小配对分数
    static constexpr float OPTIMAL_PAIR_SCORE = 0.85f;

    // 新增配对约束常量
    static constexpr float MAX_PAIR_LEN_RATIO = 0.7f;       // 最小长度比
    static constexpr float MAX_PAIR_HEIGHT_RATIO = 0.6f;    // 增大高度差容差
    static constexpr float MAX_PAIR_ANGLE_DIFF = 15.0f;     // 角度容差
    static constexpr float MIN_PAIR_DIST_RATIO = 1.8f;      // 最小距离比
    static constexpr float MAX_PAIR_DIST_RATIO = 3.0f;      // 最大距离比
    static constexpr float EXPECT_PAIR_DIST_RATIO = 2.2f;   // 期望距离比
    static constexpr float MIN_VELOCITY_COSINE = 0.95f;    // 最小速度方向相似度

    // 结构体定义
    struct LightBar {
        RotatedRect rect;
        float brightness;
        Vec3b color;
        Point2f top_point;    // 添加端点
        Point2f bottom_point; // 添加端点
        float length;         // 添加灯带长度
        int id = 0;          // 唯一标识符
        int matched_id = 0;  // 匹配对象的ID
        int frame_count = 0; // 连续出现的帧数
        bool is_matched = false; // 是否已匹配
        KalmanFilter kf;     // 卡尔曼滤波器
        mutable Point2f predicted_position; // 预测位置
        Point2f predicted_top;     // 预测的顶端位置
        Point2f predicted_bottom;  // 预测的底端位置
        Point2f velocity_top;      // 顶端速度
        Point2f velocity_bottom;   // 底端速度
        Mat predicted_state; // 添加完整状态预测
        Mat last_measurement; // 添加上一帧测量
        bool state_initialized = false; // 状态初始化标记
        
        LightBar(const RotatedRect& r, float b, const Scalar& c) 
            : rect(r), brightness(b), color(Vec3b(c[0], c[1], c[2])) {
            // 修改初始化
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
            
            // 修改卡尔曼滤波器初始化
            kf.init(8, 4, 0);
            LightBarDetector::setupKalmanFilter(kf);
            
            // 初始化状态向量
            kf.statePost.at<float>(0) = top_point.x;
            kf.statePost.at<float>(1) = top_point.y;
            kf.statePost.at<float>(4) = bottom_point.x;
            kf.statePost.at<float>(5) = bottom_point.y;
            
            predicted_top = top_point;
            predicted_bottom = bottom_point;
            velocity_top = Point2f(0,0);
            velocity_bottom = Point2f(0,0);
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

    struct BindHistory {
        ArmorPair pair;
        int frame_count;
        int unbind_count;
        ArmorPair alternative_pair;  // 替代绑定方案
        int alternative_count;       // 替代方案出现次数
        
        BindHistory(const ArmorPair& p) 
            : pair(p), frame_count(0), unbind_count(0), alternative_count(0) {}
            
        float getUnbindRate() const {
            return frame_count > 0 ? (float)unbind_count / frame_count : 0.0f;
        }

        float getAlternativeRate() const {
            return frame_count > 0 ? (float)alternative_count / frame_count : 0.0f;
        }
    };

    vector<BindHistory> binding_history;

    // 静态辅助函数
    static void setupKalmanFilter(KalmanFilter& kf);
    
    // 私有成员函数
    void predictLightBar(LightBar& light);
    float calcMatchScore(const LightBar& curr, const LightBar& prev);
    float calcMotionDeviation(const ArmorPair& p1, const ArmorPair& p2);
    float calcMotionSimilarity(const ArmorPair& p1, const ArmorPair& p2);
    bool shouldUnbind(const ArmorPair& curr, const ArmorPair& prev);
    bool canRebind(const ArmorPair& curr, const ArmorPair& target);
    bool isSimilarPair(const ArmorPair& p1, const ArmorPair& p2, float threshold = 10.0f);
    bool isSameBinding(const ArmorPair& p1, const ArmorPair& p2, float threshold = 10.0f);
    bool matchLightBars(const LightBar& curr, const LightBar& prev);
    void updateVelocity(LightBar& light, const Point2f& curr_pos, float dt = 1.0f/30.0f);
    void updateLightBarsState(vector<LightBar>& lights);

    // 新增装甲板配对结构
    struct PairCandidate {
        size_t left_idx;
        size_t right_idx;
        float score;
        float length_ratio;
        float height_diff_ratio;
        float angle_diff;
        float motion_score;
        
        PairCandidate(size_t l, size_t r) : left_idx(l), right_idx(r) {}
        bool operator<(const PairCandidate& other) const { return score > other.score; }
    };

    struct PairScore {
        float geometric_score;   // 几何特征得分
        float motion_score;      // 运动特征得分
        float tracking_score;    // 跟踪稳定性得分
        float final_score;       // 最终得分
        
        bool operator<(const PairScore& other) const {
            return final_score > other.final_score;
        }
    };

    PairScore calcPairScore(const LightBar& left, const LightBar& right) {
        PairScore score = {0, 0, 0, 0};
        
        // 1. 计算几何特征
        float len_ratio = min(left.length, right.length) / 
                         max(left.length, right.length);
        float height_diff = abs(left.rect.center.y - right.rect.center.y) /
                          ((left.length + right.length) * 0.5f);
        float angle_diff = abs(left.rect.angle - right.rect.angle);
        angle_diff = min(angle_diff, 180.0f - angle_diff);
        
        float dist = norm(right.rect.center - left.rect.center);
        float avg_length = (left.length + right.length) * 0.5f;
        float dist_ratio = dist / avg_length;

        // 2. 硬性约束检查
        if(len_ratio < MAX_PAIR_LEN_RATIO || angle_diff > MAX_PAIR_ANGLE_DIFF ||
           dist_ratio < MIN_PAIR_DIST_RATIO || dist_ratio > MAX_PAIR_DIST_RATIO) {
            return score;
        }

        // 3. 评分计算
        float length_score = pow(len_ratio, 0.5f);
        float dist_ratio_score = exp(-pow(dist_ratio - EXPECT_PAIR_DIST_RATIO, 2) / 0.5f);
        float height_score = exp(-pow(height_diff / MAX_PAIR_HEIGHT_RATIO, 2));
        float angle_score = cos(angle_diff * CV_PI / 180.0f);

        // 4. 几何总分计算
        score.geometric_score = (
            length_score * 0.35f +
            dist_ratio_score * 0.35f +
            height_score * 0.2f +
            angle_score * 0.1f
        );

        // 5. 计算运动和跟踪得分
        score.motion_score = 1.0f;  // 默认给满分
        score.tracking_score = 0.0f; // 初始帧给0分

        // 6. 最终得分
        score.final_score = score.geometric_score;  // 仅使用几何得分
        
        // 7. 调试输出
        cout << "\n详细得分分析:" << endl
             << "  长度比: " << len_ratio << " -> 得分: " << length_score << endl
             << "  距离比: " << dist_ratio << " -> 得分: " << dist_ratio_score << endl
             << "  高度差: " << height_diff << "px -> 得分: " << height_score << endl
             << "  角度差: " << angle_diff << "度 -> 得分: " << angle_score << endl
             << "  总几何得分: " << score.geometric_score << endl;

        return score;
    }

    vector<ArmorPair> matchLightBars(const vector<LightBar>& lights) {
        cout << "\n============ 装甲板匹配详细信息 ============" << endl;
        cout << "当前帧检测到灯条数量: " << lights.size() << endl;
        
        // 输出每个灯条的详细信息
        cout << "\n===== 灯条原始数据 =====" << endl;
        for(size_t i = 0; i < lights.size(); i++) {
            const auto& light = lights[i];
            cout << "\n灯条" << i << "详细数据:" << endl;
            cout << "基本属性:" << endl;
            cout << "  长度: " << light.length << "px" << endl;
            cout << "  宽度: " << min(light.rect.size.width, light.rect.size.height) << "px" << endl;
            cout << "  长宽比: " << max(light.rect.size.width, light.rect.size.height) / 
                                  min(light.rect.size.width, light.rect.size.height) << endl;
            cout << "  角度: " << light.rect.angle << "度" << endl;
            cout << "  亮度: " << light.brightness << endl;
            cout << "  连续跟踪帧数: " << light.frame_count << endl;
            
            cout << "位置信息:" << endl;
            cout << "  中心点: (" << light.rect.center.x << ", " << light.rect.center.y << ")" << endl;
            cout << "  顶端点: (" << light.top_point.x << ", " << light.top_point.y << ")" << endl;
            cout << "  底端点: (" << light.bottom_point.x << ", " << light.bottom_point.y << ")" << endl;
            
            cout << "运动信息:" << endl;
            cout << "  顶端速度: (" << light.velocity_top.x << ", " << light.velocity_top.y << ")" << endl;
            cout << "  底端速度: (" << light.velocity_bottom.x << ", " << light.velocity_bottom.y << ")" << endl;
            if(light.state_initialized) {
                cout << "  预测位置 - 顶端: (" << light.predicted_top.x << ", " << light.predicted_top.y << ")" << endl;
                cout << "  预测位置 - 底端: (" << light.predicted_bottom.x << ", " << light.predicted_bottom.y << ")" << endl;
            }
        }
        
        cout << "\n===== 配对过程详细信息 =====" << endl;
        vector<tuple<size_t, size_t, PairScore, float, float>> detailed_pairs;
        
        // 生成配对并计算详细数据
        for(size_t i = 0; i < lights.size(); i++) {
            for(size_t j = i + 1; j < lights.size(); j++) {
                const auto& left = lights[i];
                const auto& right = lights[j];
                
                // 计算原始几何特征
                float dist = norm(right.rect.center - left.rect.center);
                float avg_length = (left.length + right.length) * 0.5f;
                float dist_ratio = dist / avg_length;
                float len_ratio = min(left.length, right.length) / 
                                max(left.length, right.length);
                float height_diff = abs(left.rect.center.y - right.rect.center.y);
                float rel_height_diff = height_diff / avg_length;
                float angle_diff = abs(left.rect.angle - right.rect.angle);
                angle_diff = min(angle_diff, 180.0f - angle_diff);
                float top_dist = norm(left.top_point - right.top_point);
                float bottom_dist = norm(left.bottom_point - right.bottom_point);
                
                cout << "\n配对(" << i << "," << j << ")完整数据:" << endl;
                cout << "距离数据:" << endl;
                cout << "  中心点距离: " << dist << "px" << endl;
                cout << "  顶端距离: " << top_dist << "px" << endl;
                cout << "  底端距离: " << bottom_dist << "px" << endl;
                cout << "  平均长度: " << avg_length << "px" << endl;
                cout << "  距离比(中心距离/平均长度): " << dist_ratio << endl;
                
                cout << "形态特征:" << endl;
                cout << "  长度比(短/长): " << len_ratio << endl;
                cout << "  高度差: " << height_diff << "px" << endl;
                cout << "  相对高度差(高度差/平均长度): " << rel_height_diff << endl;
                cout << "  角度差: " << angle_diff << "度" << endl;
                
                // 计算运动相似度
                if(left.frame_count > 1 && right.frame_count > 1) {
                    float vel_cos_top = left.velocity_top.dot(right.velocity_top) /
                                      (norm(left.velocity_top) * norm(right.velocity_top) + 1e-6);
                    float vel_cos_bottom = left.velocity_bottom.dot(right.velocity_bottom) /
                                         (norm(left.velocity_bottom) * norm(right.velocity_bottom) + 1e-6);
                    cout << "运动特征:" << endl;
                    cout << "  顶端速度夹角余弦: " << vel_cos_top << endl;
                    cout << "  底端速度夹角余弦: " << vel_cos_bottom << endl;
                    cout << "  左灯条速度: " << norm(left.velocity_top) << "px/s" << endl;
                    cout << "  右灯条速度: " << norm(right.velocity_top) << "px/s" << endl;
                }

                // ...existing score calculation code...
                PairScore score = calcPairScore(lights[i], lights[j]);
                
                // 只有在距离比合理范围内才考虑配对
                if(dist_ratio >= 1.8f && dist_ratio <= 2.8f) {
                    detailed_pairs.emplace_back(i, j, score, dist_ratio, rel_height_diff);
                    cout << "评分详情:" << endl;
                    cout << "  几何得分: " << score.geometric_score << endl;
                    cout << "  运动得分: " << score.motion_score << endl;
                    cout << "  跟踪得分: " << score.tracking_score << endl;
                    cout << "  最终得分: " << score.final_score << endl;
                } else {
                    cout << "配对失败: 距离比(" << dist_ratio << ")超出范围[1.8, 2.8]" << endl;
                }
            }
        }

        // ...existing matching logic...
        vector<ArmorPair> pairs;
        vector<bool> used(lights.size(), false);
        vector<tuple<size_t, size_t, PairScore>> candidates;
        
        cout << "\n===== 原始灯条信息 =====" << endl;
        for(size_t i = 0; i < lights.size(); i++) {
            const auto& light = lights[i];
            cout << "灯条" << i << ":" << endl;
            cout << "  长度: " << light.length << "px" << endl;
            cout << "  角度: " << light.rect.angle << "度" << endl;
            cout << "  中心点: (" << light.rect.center.x << ", " << light.rect.center.y << ")" << endl;
            cout << "  亮度: " << light.brightness << endl;
            cout << "  颜色: [" << (int)light.color[0] << "," 
                 << (int)light.color[1] << "," << (int)light.color[2] << "]" << endl;
        }
        
        cout << "\n===== 所有可能的配对 =====" << endl;
        // 1. 生成所有可能的配对并排序
        for(size_t i = 0; i < lights.size(); i++) {
            for(size_t j = i + 1; j < lights.size(); j++) {
                // 计算原始几何特征
                float dist = norm(lights[i].rect.center - lights[j].rect.center);
                float avg_length = (lights[i].length + lights[j].length) * 0.5f;
                float dist_ratio = dist / avg_length;
                float len_ratio = min(lights[i].length, lights[j].length) / 
                                max(lights[i].length, lights[j].length);
                float height_diff = abs(lights[i].rect.center.y - lights[j].rect.center.y);
                float angle_diff = abs(lights[i].rect.angle - lights[j].rect.angle);
                angle_diff = min(angle_diff, 180.0f - angle_diff);
                
                cout << "\n配对(" << i << "," << j << ")原始数据:" << endl;
                cout << "  距离: " << dist << "px" << endl;
                cout << "  平均长度: " << avg_length << "px" << endl;
                cout << "  距离比: " << dist_ratio << endl;
                cout << "  长度比: " << len_ratio << endl;
                cout << "  高度差: " << height_diff << "px" << endl;
                cout << "  角度差: " << angle_diff << "度" << endl;
                
                // 计算得分
                PairScore score = calcPairScore(lights[i], lights[j]);
                
                // 只有在距离比合理范围内才考虑配对
                if(dist_ratio >= 1.8f && dist_ratio <= 2.8f) {
                    candidates.emplace_back(i, j, score);
                    
                    cout << "  配对评分:" << endl;
                    cout << "    几何得分: " << score.geometric_score << endl;
                    cout << "    运动得分: " << score.motion_score << endl;
                    cout << "    跟踪得分: " << score.tracking_score << endl;
                    cout << "    最终得分: " << score.final_score << endl;
                } else {
                    cout << "  距离比不符合要求,配对失败" << endl;
                }
            }
        }

        // ...排序和选择逻辑保持不变...
        sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) {
                return get<2>(a).final_score > get<2>(b).final_score;
            });

        cout << "\n===== 最终选择的配对 =====" << endl;
        int required_pairs = (lights.size() >= 4) ? 2 : 1;
        int selected_pairs = 0;

        // 第一轮选择
        for(const auto& [i, j, score] : candidates) {
            if(selected_pairs >= required_pairs) break;
            if(!used[i] && !used[j] && score.geometric_score > MIN_PAIR_SCORE) { // 使用geometric_score
                pairs.emplace_back(lights[i], lights[j]);
                used[i] = used[j] = true;
                selected_pairs++;
                
                cout << "\n选择配对" << selected_pairs << ":" << endl;
                cout << "  左灯条: " << i << ", 右灯条: " << j << endl;
                cout << "  评分: " << score.final_score << endl;
                cout << "  几何得分: " << score.geometric_score << endl;
                cout << "  运动得分: " << score.motion_score << endl;
                cout << "  跟踪得分: " << score.tracking_score << endl;
            }
        }
        
        // ...第二轮选择保持不变...
        
        cout << "\n===== 匹配统计信息 =====" << endl;
        cout << "  尝试配对总数: " << lights.size() * (lights.size() - 1) / 2 << endl;
        cout << "  合格配对数量: " << candidates.size() << endl;
        cout << "  期望装甲板数: " << required_pairs << endl;
        cout << "  实际匹配数量: " << pairs.size() << endl;
        cout << "  匹配成功率: " << (float)pairs.size() / required_pairs * 100 << "%" << endl;
        
        if(!candidates.empty()) {
            float avg_score = 0;
            for(size_t i = 0; i < pairs.size(); i++) {
                avg_score += get<2>(candidates[i]).final_score;
            }
            avg_score /= pairs.size();
            cout << "  平均配对分数: " << avg_score << endl;
        }
        cout << "================================" << endl;
        
        cout << "\n============ 最终配对结果 ============" << endl;
        cout << "尝试配对总数: " << lights.size() * (lights.size() - 1) / 2 << endl;
        cout << "有效配对数量: " << detailed_pairs.size() << endl;
        cout << "实际匹配数量: " << pairs.size() << endl;
        
        if(!pairs.empty()) {
            for(size_t i = 0; i < pairs.size(); i++) {
                const auto& pair = pairs[i];
                cout << "\n成功配对 #" << i+1 << ":" << endl;
                cout << "  左灯条中心: (" << pair.left.rect.center.x << ", " 
                     << pair.left.rect.center.y << ")" << endl;
                cout << "  右灯条中心: (" << pair.right.rect.center.x << ", " 
                     << pair.right.rect.center.y << ")" << endl;
                cout << "  装甲板中心: (" << (pair.left.rect.center.x + pair.right.rect.center.x)/2 << ", "
                     << (pair.left.rect.center.y + pair.right.rect.center.y)/2 << ")" << endl;
                float dist = norm(pair.right.rect.center - pair.left.rect.center);
                float avg_len = (pair.left.length + pair.right.length) * 0.5f;
                cout << "  距离比: " << dist/avg_len << endl;
                cout << "  装甲板宽度: " << dist << "px" << endl;
                cout << "  装甲板高度: " << avg_len << "px" << endl;
            }
        }
        cout << "=======================================" << endl;
        
        return pairs;
    }

public:
    LightBarDetector() = default;
    const Mat& getDebugImage() const { return debug_img; }
    void showDebugInfo(const Mat& frame, const vector<LightBar>& lights);
    vector<LightBar> detect(Mat& frame);
};

// 实现静态函数
void LightBarDetector::setupKalmanFilter(KalmanFilter& kf) {
    kf.init(8, 4, 0);
    float dt = 1.0f/30.0f;
    
    // 修改状态转移矩阵 - 调整速度衰减
    float decay = 0.95f;  // 降低衰减速率
    kf.transitionMatrix = (Mat_<float>(8, 8) <<
        1,0,dt,0,0,0,0,0,
        0,1,0,dt,0,0,0,0,
        0,0,decay,0,0,0,0,0,
        0,0,0,decay,0,0,0,0,
        0,0,0,0,1,0,dt,0,
        0,0,0,0,0,1,0,dt,
        0,0,0,0,0,0,decay,0,
        0,0,0,0,0,0,0,decay);

    // 降低过程噪声
    float pos_noise = 0.1f;  // 降低位置噪声
    float vel_noise = 1.0f;  // 降低速度噪声
    
    kf.processNoiseCov = Mat::eye(8, 8, CV_32F);
    for(int i = 0; i < 8; i++) {
        kf.processNoiseCov.at<float>(i,i) = (i % 2 == 0) ? pos_noise : vel_noise;
    }

    // 提高测量置信度
    setIdentity(kf.measurementNoiseCov, Scalar::all(0.005));
    
    // 降低初始状态协方差
    kf.errorCovPost = Mat::eye(8, 8, CV_32F);
    for(int i = 0; i < 8; i++) {
        kf.errorCovPost.at<float>(i,i) = 0.1f;
    }

    // 设置测量矩阵
    kf.measurementMatrix = Mat::zeros(4, 8, CV_32F);
    kf.measurementMatrix.at<float>(0,0) = 1;  // x位置
    kf.measurementMatrix.at<float>(1,1) = 1;  // y位置
    kf.measurementMatrix.at<float>(2,4) = 1;  // x位置(底端)
    kf.measurementMatrix.at<float>(3,5) = 1;  // y位置(底端)
}

// 实现其他成员函数
void LightBarDetector::predictLightBar(LightBar& light) {
    if(!light.state_initialized) {
        light.kf.statePost.setTo(0);
        light.kf.statePost.at<float>(0) = light.top_point.x;
        light.kf.statePost.at<float>(1) = light.top_point.y;
        light.kf.statePost.at<float>(4) = light.bottom_point.x;
        light.kf.statePost.at<float>(5) = light.bottom_point.y;
        light.state_initialized = true;
        light.last_measurement = (Mat_<float>(4,1) << 
            light.top_point.x, light.top_point.y,
            light.bottom_point.x, light.bottom_point.y);
        return;
    }

    Mat prediction = light.kf.predict();
    light.predicted_state = prediction.clone();
    
    Mat_<float> measurement(4, 1);
    measurement(0) = light.top_point.x;
    measurement(1) = light.top_point.y;
    measurement(2) = light.bottom_point.x;
    measurement(3) = light.bottom_point.y;

    // 计算测量增量
    Mat delta_measurement = measurement - light.last_measurement;
    light.last_measurement = measurement.clone();

    // 更新速度估计
    if(light.frame_count > 1) {
        light.velocity_top = Point2f(delta_measurement.at<float>(0), 
                                   delta_measurement.at<float>(1)) * 30.0f;
        light.velocity_bottom = Point2f(delta_measurement.at<float>(2),
                                      delta_measurement.at<float>(3)) * 30.0f;
    }

    Mat corrected = light.kf.correct(measurement);

    // 保存修正结果
    light.predicted_top = Point2f(corrected.at<float>(0), corrected.at<float>(1));
    light.predicted_bottom = Point2f(corrected.at<float>(4), corrected.at<float>(5));

    // 记录调试信息
    cout << "\n===== 卡尔曼滤波调试信息 =====" << endl;
    cout << "灯条ID: " << light.id << ", 连续跟踪帧数: " << light.frame_count << endl;
    
    // 输出预测误差
    float top_pred_error = norm(Point2f(prediction.at<float>(0), prediction.at<float>(1)) - light.top_point);
    float bottom_pred_error = norm(Point2f(prediction.at<float>(4), prediction.at<float>(5)) - light.bottom_point);
    cout << "预测误差 - 顶点: " << top_pred_error << "px, 底点: " << bottom_pred_error << "px" << endl;
    
    // 输出速度信息
    Point2f top_velocity(prediction.at<float>(2), prediction.at<float>(3));
    Point2f bottom_velocity(prediction.at<float>(6), prediction.at<float>(7));
    cout << "速度向量 - 顶点: (" << top_velocity.x << ", " << top_velocity.y 
         << "), 底点: (" << bottom_velocity.x << ", " << bottom_velocity.y << ")" << endl;
    
    // 输出协方差矩阵对角线元素(不确定度)
    cout << "状态不确定度:" << endl;
    for(int i = 0; i < 8; i++) {
        cout << "  状态" << i << ": " << light.kf.errorCovPost.at<float>(i,i) << endl;
    }
    
    // 计算预测稳定性指标
    float stability_score = 1.0f / (1.0f + (top_pred_error + bottom_pred_error) * 0.5f);
    cout << "预测稳定性得分(0-1): " << stability_score << endl;

    // 保存预测结果
    light.predicted_top = Point2f(prediction.at<float>(0), prediction.at<float>(1));
    light.predicted_bottom = Point2f(prediction.at<float>(4), prediction.at<float>(5));
    light.velocity_top = Point2f(corrected.at<float>(2), corrected.at<float>(3));
    light.velocity_bottom = Point2f(corrected.at<float>(6), corrected.at<float>(7));

    // 输出实际修正后的位置
    cout << "修正后位置 - 顶点: (" << light.predicted_top.x << ", " << light.predicted_top.y 
         << "), 底点: (" << light.predicted_bottom.x << ", " << light.predicted_bottom.y << ")" << endl;
    cout << "================================" << endl;

    // 记录调试信息时增加速度预测评估
    if(light.frame_count > 1) {
        cout << "速度预测评估：" << endl;
        cout << "  速度变化率(顶点): " 
             << norm(light.velocity_top - Point2f(prediction.at<float>(2), prediction.at<float>(3))) << endl;
        cout << "  速度变化率(底点): " 
             << norm(light.velocity_bottom - Point2f(prediction.at<float>(6), prediction.at<float>(7))) << endl;
    }
}

float LightBarDetector::calcMatchScore(const LightBar& curr, const LightBar& prev) {
    // 卡尔曼预测分数
    float pred_score = 0.0f;
    if(prev.frame_count > 0) {
        Point2f pred_top = prev.predicted_top + prev.velocity_top * (1.0f/30.0f);
        Point2f pred_bottom = prev.predicted_bottom + prev.velocity_bottom * (1.0f/30.0f);
        float pos_error = (norm(curr.top_point - pred_top) + 
                         norm(curr.bottom_point - pred_bottom)) * 0.5f;
        pred_score = 1.0f / (1.0f + pos_error * 0.1f);
    }

    // 几何特征分数
    float length_ratio = min(curr.length, prev.length) / max(curr.length, prev.length);
    float angle_diff = abs(curr.rect.angle - prev.rect.angle);
    angle_diff = min(angle_diff, 180.0f - angle_diff);
    float angle_score = 1.0f - angle_diff / 180.0f;
    
    // 速度连续性分数
    float motion_score = 1.0f;
    if(prev.frame_count > 1) {
        float vel_similarity = prev.velocity_top.dot(curr.velocity_top) / 
                             (norm(prev.velocity_top) * norm(curr.velocity_top) + 1e-6);
        motion_score = (vel_similarity + 1.0f) * 0.5f;
    }

    // 动态权重
    float pred_weight = prev.frame_count > 0 ? 0.5f : 0.0f;
    float geom_weight = 0.3f;
    float motion_weight = prev.frame_count > 1 ? 0.2f : 0.0f;
    float remain_weight = 1.0f - (pred_weight + geom_weight + motion_weight);

    return pred_score * pred_weight +
           (length_ratio * 0.6f + angle_score * 0.4f) * geom_weight +
           motion_score * motion_weight +
           (1.0f / (1.0f + norm(curr.rect.center - prev.rect.center))) * remain_weight;
}

float LightBarDetector::calcMotionDeviation(const ArmorPair& p1, const ArmorPair& p2) {
    Point2f vec1 = p1.right.rect.center - p1.left.rect.center;
    Point2f vec2 = p2.right.rect.center - p2.left.rect.center;
    float angle_diff = abs(atan2(vec1.y, vec1.x) - atan2(vec2.y, vec2.x));
    float length_diff = abs(norm(vec1) - norm(vec2)) / max(norm(vec1), norm(vec2));
    return angle_diff + length_diff;  // 综合角度和长度差异
}

float LightBarDetector::calcMotionSimilarity(const ArmorPair& p1, const ArmorPair& p2) {
    Point2f vec1 = p1.right.rect.center - p1.left.rect.center;
    Point2f vec2 = p2.right.rect.center - p1.left.rect.center;
    float angle_sim = cos(atan2(vec1.y, vec1.x) - atan2(vec2.y, vec2.x));
    float length_sim = min(norm(vec1), norm(vec2)) / max(norm(vec1), norm(vec2));
    return 0.95f * angle_sim + 0.05f * length_sim;  // 大幅提高角度相似度的权重
}

bool LightBarDetector::shouldUnbind(const ArmorPair& curr, const ArmorPair& prev) {
    return calcMotionDeviation(curr, prev) > MOTION_DEVIATION_THRESHOLD;
}

bool LightBarDetector::canRebind(const ArmorPair& curr, const ArmorPair& target) {
    return calcMotionSimilarity(curr, target) > MOTION_SIMILARITY_THRESHOLD;
}

bool LightBarDetector::isSimilarPair(const ArmorPair& p1, const ArmorPair& p2, float threshold) {
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
    if(ratio1 < 1.8f || ratio1 > 2.85f || ratio2 < 1.8f || ratio2 > 2.85f) {
        return false;
    }
    
    // 先计算中点连线角度
    float midAngle1 = atan2(p1.right.rect.center.y - p1.left.rect.center.y, 
                           p1.right.rect.center.x - p1.left.rect.center.x);
    float midAngle2 = atan2(p2.right.rect.center.y - p2.left.rect.center.y, 
                           p2.right.rect.center.x - p2.left.rect.center.x);
    
    // 根据角度动态调整高度差阈值
    float angleAbs1 = abs(midAngle1);
    float angleAbs2 = abs(midAngle2);
    float heightThreshold1, heightThreshold2;
    
    // 180到150度之间时角度越小阈值越大
    if(angleAbs1 > 2.61799f && angleAbs1 <= 3.14159f) { // 150到180度
        heightThreshold1 = 2.0f + (3.14159f - angleAbs1) * 0.5f; // 角度越小阈值越大
    } else {
        heightThreshold1 = 2.0f; // 小于150度时固定为2.0
    }
    
    if(angleAbs2 > 2.61799f && angleAbs2 <= 3.14159f) { // 150到180度
        heightThreshold2 = 2.0f + (3.14159f - angleAbs2) * 0.5f; // 角度越小阈值越大
    } else {
        heightThreshold2 = 2.0f; // 小于150度时固定为2.0
    }
    
    // 计算高度差比值
    float avgLen1 = (norm(p1.left.top_point - p1.left.bottom_point) + norm(p1.right.top_point - p1.right.bottom_point)) / 2;
    float avgLen2 = (norm(p2.left.top_point - p2.left.bottom_point) + norm(p2.right.top_point - p2.right.bottom_point)) / 2;
    float heightDiffRatio1 = abs(p1.left.rect.center.y - p1.right.rect.center.y) / avgLen1;
    float heightDiffRatio2 = abs(p2.left.rect.center.y - p2.right.rect.center.y) / avgLen2;
    
    if(heightDiffRatio1 > heightThreshold1 || heightDiffRatio2 > heightThreshold2) {
        
        return false;
    }
    
    // 角度差不超过15度
    float angle1 = atan2(p1.right.rect.center.y - p1.left.rect.center.y, 
                        p1.right.rect.center.x - p1.left.rect.center.x);
    float angle2 = atan2(p2.right.rect.center.y - p2.left.rect.center.y, 
                        p2.right.rect.center.x - p2.left.rect.center.x);
    if(abs(angle1 - angle2) > CV_PI/6) { // 30度
        
        return false;
    }
    
    // 中点连线角度不超过60度
    if((midAngle1 > -CV_PI*2/3 && midAngle1 < -CV_PI/3) || (midAngle1 > CV_PI/3 && midAngle1 < CV_PI*2/3) || 
       (midAngle2 > -CV_PI*2/3 && midAngle2 < -CV_PI/3) || (midAngle2 > CV_PI/3 && midAngle2 < CV_PI*2/3)) { // -120到-180度和120到180度
        
        return false;
    }
    
    // 中心点距离阈值
    Point2f center1 = p1.left.rect.center;
    Point2f center2 = p2.left.rect.center;
    return norm(center1 - center2) < threshold;
}

bool LightBarDetector::isSameBinding(const ArmorPair& p1, const ArmorPair& p2, float threshold) {
    return isSimilarPair(p1, p2) && 
           norm(p1.left.rect.center - p2.left.rect.center) < threshold &&
           norm(p1.right.rect.center - p2.right.rect.center) < threshold;
}

bool LightBarDetector::matchLightBars(const LightBar& curr, const LightBar& prev) {
    float center_dist = norm(curr.rect.center - prev.rect.center);
    if(center_dist > MAX_MATCH_DISTANCE) return false;
    
    // 改进角度差计算
    float angle_diff = abs(curr.rect.angle - prev.rect.angle);
    angle_diff = min(angle_diff, 180.0f - angle_diff);
    if(angle_diff > MAX_ANGLE_DIFF) return false;
    
    // 添加尺寸一致性检查
    float size_ratio = min(curr.length, prev.length) / max(curr.length, prev.length);
    if(size_ratio < 0.8f) return false;  // 提高尺寸一致性要求

    // 改进运动预测评分
    float motion_score = 1.0f;
    if(prev.frame_count > 1) {
        Point2f predicted_pos = prev.predicted_top + prev.velocity_top * (1.0f/30.0f);
        float pred_error = norm(curr.top_point - predicted_pos);
        float vel_consistency = norm(curr.velocity_top - prev.velocity_top);
        
        motion_score = 1.0f / (1.0f + pred_error * 0.05f + vel_consistency * 0.1f);
    }

    // 修改匹配分数计算
    float angle_score = 1.0f - angle_diff / MAX_ANGLE_DIFF;
    float dist_score = 1.0f / (1.0f + center_dist * 0.1f);
    float match_score = angle_score * 0.3f + 
                      dist_score * 0.4f + 
                      size_ratio * 0.2f +
                      motion_score * 0.1f;

    // 根据历史帧数调整阈值
    float dynamic_threshold = MIN_TRACK_QUALITY;
    if(prev.frame_count > 1) {
        dynamic_threshold *= (0.8f + 0.2f / log(prev.frame_count + 1));
    }

    return match_score > dynamic_threshold;
}

void LightBarDetector::updateVelocity(LightBar& light, const Point2f& curr_pos, float dt) {
    if(light.frame_count > 1) {
        Point2f measured_vel = (curr_pos - light.predicted_position) / dt;
        // 自适应EMA系数
        float alpha = 0.3f + 0.4f * exp(-light.frame_count * 0.1f);
        light.velocity_top = light.velocity_top * (1.0f - alpha) + measured_vel * alpha;
        
        // 速度限幅
        float max_speed = 500.0f;  // 像素/秒
        float speed = norm(light.velocity_top);
        if(speed > max_speed) {
            light.velocity_top *= (max_speed / speed);
        }
    }
}

void LightBarDetector::updateLightBarsState(vector<LightBar>& lights) {
    static vector<LightBar> prev_lights;
    // ... 原有的状态更新代码 ...
}

void LightBarDetector::showDebugInfo(const Mat& frame, const vector<LightBar>& lights) {
    debug_img = frame.clone();
    
    // 匹配装甲板
    auto armor_pairs = matchLightBars(lights);
    
    // 显示灯条端点和装甲板连接
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
    
    // 绘制装甲板配对连接线
    for(const auto& pair : armor_pairs) {
        line(debug_img, 
             pair.left.rect.center, 
             pair.right.rect.center, 
             Scalar(0,255,0), 2);
    }
    
    imshow("Debug Window", debug_img);
}

vector<LightBarDetector::LightBar> LightBarDetector::detect(Mat& frame) {
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
            imwrite("light_bars.jpg", detector.getDebugImage());
        }
    }
    
    destroyAllWindows();
    return 0;
}