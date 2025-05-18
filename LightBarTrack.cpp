#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
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
    int morph_open_size = 3;
    int dilate_iterations = 2;
    float min_aspect_ratio = 2.0f;   
    float max_aspect_ratio = 100.0f;  
};

class LightBarDetector {
private:
    DetectionParams params;
    Mat all_rect_debug;  // 添加用于显示所有矩形的调试图像
    
    // 相机标定参数
    Mat camera_matrix;
    Mat dist_coeffs;
    Mat map1, map2;
    
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
                              camera_matrix, Size(630, 480), CV_16SC2, map1, map2);
    }
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
        
        bool isBlue(const Mat& frame, const Rect& bbox) {
            // 计算膨胀50%后的区域
            float scale = 1.2f;
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
            Scalar lower_blue(91, 193, 97);
            Scalar upper_blue(141, 255, 255);
            
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
            return blue_ratio > 0.05; // 蓝色像素占比阈值已设为50%
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
        
     
        float ratio1 = getRatio(p1.left, p1.right), ratio2 = getRatio(p2.left, p2.right);
        if(ratio1 < 3.5f || ratio1 > 6.0f || ratio2 < 3.5f || ratio2 > 6.0f) {
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

    

public:
    LightBarDetector() {
        params = DetectionParams();
        initCameraParams();  // 构造函数中初始化相机参数
    }
    
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
            

            // 计算中心点
            Point2f center(0,0);
            for(const auto& p : corners) center += p;
            center *= (1.0 / corners.size());
            
            // 按相对于中心点的极角排序
            auto compare_angle = [center](const Point2f& a, const Point2f& b) {
                Point2f da = a - center;
                Point2f db = b - center;
                return atan2(da.y, da.x) < atan2(db.y, db.x);
            };
            sort(corners.begin(), corners.end(), compare_angle);
            
            // 确保第一个点是左上角点
            int top_left_idx = 0;
            for(int i = 1; i < 4; i++) {
                if(corners[i].x + corners[i].y < corners[top_left_idx].x + corners[top_left_idx].y) {
                    top_left_idx = i;
                }
            }
            rotate(corners.begin(), corners.begin() + top_left_idx, corners.end());
            
            // 保留PnP解算功能但不绘制额外信息
            vector<Point3f> armor_3d_points = {
                Point3f(-12.25f, -3.0f, 0),  // 左上(单位:cm)
                Point3f(12.25f, -3.0f, 0),   // 右上(单位:cm)
                Point3f(12.25f, 3.0f, 0),    // 右下(单位:cm)
                Point3f(-12.25f, 3.0f, 0)    // 左下(单位:cm)
            };
            
            // PnP解算
            Mat rvec, tvec;
            bool pnpSuccess = solvePnP(armor_3d_points, corners, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_IPPE);
            if(!pnpSuccess) {
                // 如果IPPE失败，尝试SQPNP
                pnpSuccess = solvePnP(armor_3d_points, corners, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_SQPNP);
                if(!pnpSuccess) {
                    cerr << "PnP解算失败: " << cv::format(camera_matrix, cv::Formatter::FMT_DEFAULT) << endl;
                    return;
                }
            }
            
            // 计算重投影误差
            vector<Point2f> reprojected_points;
            projectPoints(armor_3d_points, rvec, tvec, camera_matrix, dist_coeffs, reprojected_points);
            double total_error = 0.0;
            for(size_t i = 0; i < corners.size(); ++i) {
                double error = norm(reprojected_points[i] - corners[i]);
                total_error += error;
                // cout << "角点" << i << " 重投影误差: " << error << " 像素" << endl;
            }
            // cout << "平均重投影误差: " << total_error/corners.size() << " 像素" << endl;
            
            // 计算并输出四个角点的三维坐标
            vector<Point3f> world_corners(4);
            Mat R;
            Rodrigues(rvec, R);
            for(int i = 0; i < 4; i++) {
                // 根据装甲板3D模型几何关系和PnP解算结果计算角点坐标
                Mat point_mat = (Mat_<double>(3,1) << armor_3d_points[i].x, armor_3d_points[i].y, armor_3d_points[i].z);
                Mat rotated_point = R * point_mat;
                
                // 考虑装甲板实际几何关系，调整角点位置
                double scale_factor = 2; // 适当放大系数
                world_corners[i] = Point3f(
                    rotated_point.at<double>(0) * scale_factor + tvec.at<double>(0),
                    rotated_point.at<double>(1) * scale_factor + tvec.at<double>(1),
                    rotated_point.at<double>(2) * scale_factor + tvec.at<double>(2)
                );
                // cout << "角点" << i << " 3D坐标: (" 
//                      << world_corners[i].x << ", " 
//                      << world_corners[i].y << ", " 
//                      << world_corners[i].z << ")" << endl;
            }
            
            // 计算装甲板中点三维坐标
            Point3f center_3d(
                (world_corners[0].x + world_corners[1].x + world_corners[2].x + world_corners[3].x) / 4,
                (world_corners[0].y + world_corners[1].y + world_corners[2].y + world_corners[3].y) / 4,
                (world_corners[0].z + world_corners[1].z + world_corners[2].z + world_corners[3].z) / 4
            );
            // cout << "装甲板中点3D坐标: (" 
            //      << center_3d.x << ", " 
            //      << center_3d.y << ", " 
            //      << center_3d.z << ")" << endl;
                 
            // 计算装甲板法向量
            Mat normal_vector_mat = R * (Mat_<double>(3, 1) << 0, 0, 1);
            Vec3d normal_vector(normal_vector_mat.at<double>(0), normal_vector_mat.at<double>(1), normal_vector_mat.at<double>(2));
            
            // 计算两个方向的法向量
            Vec3d reverse_normal_vector(-normal_vector[0], -normal_vector[1], -normal_vector[2]);
            
            // 验证法向量是否正确
            double norm_length = sqrt(normal_vector[0]*normal_vector[0] + 
                                     normal_vector[1]*normal_vector[1] + 
                                     normal_vector[2]*normal_vector[2]);
            if(abs(norm_length - 1.0) > 0.01) {
                cerr << "警告: 法向量长度不为1 (" << norm_length << ")" << endl;
                normal_vector /= norm_length;  // 归一化
            }
            
            // 输出装甲板法向量信息
            // cout << "装甲板法向量(归一化): (" 
//                  << normal_vector[0] << ", " 
//                  << normal_vector[1] << ", " 
//                  << normal_vector[2] << ")" << endl;
                
            // 通过四个角点三维坐标计算法向量
            Vec3d edge1(world_corners[1].x - world_corners[0].x, 
                       world_corners[1].y - world_corners[0].y, 
                       world_corners[1].z - world_corners[0].z);
            Vec3d edge2(world_corners[3].x - world_corners[0].x, 
                       world_corners[3].y - world_corners[0].y, 
                       world_corners[3].z - world_corners[0].z);
            Vec3d normal = edge1.cross(edge2);
            double four_point_norm_length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
            if(four_point_norm_length > 1e-6) {
                normal /= four_point_norm_length;  // 归一化
                // cout << "四点法向量(归一化): (" 
//                      << normal[0] << ", " 
//                      << normal[1] << ", " 
//                      << normal[2] << ")" << endl;
            } else {
                cerr << "警告: 四点法向量长度为零" << endl;
            }

            
            // 将装甲板中心点3D坐标转换为2D像素坐标
            vector<Point3f> center_3d_points = {center_3d};
            vector<Point2f> center_2d_points;
            // 使用零旋转向量，因为中心点已经在世界坐标系中
            Mat zero_rvec = Mat::zeros(3, 1, CV_64F);
            projectPoints(center_3d_points, zero_rvec, tvec, camera_matrix, dist_coeffs, center_2d_points);
            
            // 输出调试信息
            // cout << "中心点3D坐标: (" << center_3d.x << ", " << center_3d.y << ", " << center_3d.z << ")" << endl;
            // cout << "中心点2D坐标: (" << center_2d_points[0].x << ", " << center_2d_points[0].y << ")" << endl;
            
            // 在debug_img上绘制装甲板中心点
            circle(debug_img, center_2d_points[0], 3, Scalar(0, 255, 255), -1);
            
            // 绘制四个角点的2D坐标
            vector<Point2f> corner_2d_points;
            projectPoints(world_corners, zero_rvec, tvec, camera_matrix, dist_coeffs, corner_2d_points);
            
            // 输出角点坐标
            for(int i = 0; i < 4; i++) {
                // cout << "角点" << i << " 2D坐标: (" << corner_2d_points[i].x << ", " << corner_2d_points[i].y << ")" << endl;
                circle(debug_img, corner_2d_points[i], 3, Scalar(0, 0, 255), -1);
                line(debug_img, corner_2d_points[i], corner_2d_points[(i+1)%4], Scalar(255, 0, 0), 1);
            }
            
            // 验证坐标转换一致性
            for(int i = 0; i < 4; i++) {
                // 重新投影3D点以验证误差
                vector<Point3f> temp_3d = {world_corners[i]};
                vector<Point2f> reprojected;
                
                // 使用零旋转向量重新投影，确保与原始投影一致
                Mat zero_rvec = Mat::zeros(3, 1, CV_64F);
                projectPoints(temp_3d, zero_rvec, tvec, camera_matrix, dist_coeffs, reprojected);
                
                if(reprojected.empty()) {
                    cerr << "投影失败: " << cv::format(camera_matrix, cv::Formatter::FMT_DEFAULT) << endl;
                    continue;
                }
                float error = norm(reprojected[0] - corners[i]);
                // cout << "角点" << i << " 3D->2D误差: " << error << endl;
                
                // 输出详细投影信息用于调试
                // cout << "原始角点" << i << " 2D坐标: (" << corners[i].x << ", " << corners[i].y << ")" << endl;
                // cout << "重投影角点" << i << " 2D坐标: (" << reprojected[0].x << ", " << reprojected[0].y << ")" << endl;
                // cout << "相机矩阵: " << cv::format(camera_matrix, cv::Formatter::FMT_DEFAULT) << endl;
                // cout << "畸变系数: " << cv::format(dist_coeffs, cv::Formatter::FMT_DEFAULT) << endl;
            }
            
            // 输出装甲板中点2D坐标
            // cout << "装甲板中点2D坐标: (" 
            //      << center_2d_points[0].x << ", " 
            //      << center_2d_points[0].y << ")" << endl;
            
            // 在左上角显示中点三维坐标
            string coord_text = cv::format("point3D: (%.1f,%.1f,%.1f)", center_3d.x, center_3d.y, center_3d.z);
            putText(debug_img, coord_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        }
        
        imshow("Debug Window", debug_img);
    }

    vector<LightBar> detect(Mat& frame) {
        auto lights = vector<LightBar>();
        
        // 应用畸变校正
        Mat undistorted;
        remap(frame, undistorted, map1, map2, INTER_LINEAR);
        
        all_rect_debug = undistorted.clone();  // 初始化调试图像
        debug_img = undistorted.clone();  // 移动到函数开始处，用于绘制所有矩形
        Mat gray;
        cvtColor(undistorted, gray, COLOR_BGR2GRAY);
        
        // 1. 亮度筛选
        Mat brightMask = getBrightnessFilteredImage(gray);
        
        // 2. 二值化 
        Mat bin;
        threshold(gray, bin, params.threshold_value, 255, THRESH_BINARY);
        bin = bin & brightMask;  // 应用亮度mask
        
        // 3. 面积筛选
        Mat areaMask = getAreaFilteredImage(bin);
        bin = bin & areaMask;
        
        // 4. 交替执行腐蚀和膨胀操作10次
        Mat erode_element = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
        Mat dilate_element = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
            erode(bin, bin, erode_element);
            dilate(bin, bin, dilate_element);
         
        
        
        // 显示形态学操作结果
        imshow("Morphology Result", bin);
        
   
        
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
                
     
                
                // 绘制最小外接矩形
                if(aspect_ratio >= params.min_aspect_ratio && aspect_ratio <= params.max_aspect_ratio) {
                    Rect bbox = r.boundingRect();
        if(bbox.x >= 0 && bbox.y >= 0 && 
           bbox.x + bbox.width <= frame.cols && 
           bbox.y + bbox.height <= frame.rows) {
            LightBar bar(r, mean(gray(bbox))[0], mean(frame(bbox)));
            if(bar.isBlue(frame, bbox)) {  // 确保颜色验证被调用
                // 只在颜色验证通过后绘制灯带
                for(int i = 0; i < 4; i++) {
                    line(all_rect_debug, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 2);
                }
                lights.push_back(bar);
                // cout << "通过颜色验证的灯条" << endl;  // 添加调试信息
            } else {
                // cout << "未通过颜色验证的灯条" << endl;  // 添加调试信息
            }
        } else {
            cerr << "无效的边界框: " << bbox << endl;
        }
                    // cout << "找到一个灯带!" << endl;
                }
            } catch(...) {
                cout << "处理轮廓时出现异常" << endl;
            }
        }
        
        // cout << "最终找到 " << lights.size() << " 个灯条" << endl;
        showDebugInfo(frame, lights);
        imshow("All Rectangles", all_rect_debug);  
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
 
    namedWindow("All Rectangles", WINDOW_AUTOSIZE);  
    
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