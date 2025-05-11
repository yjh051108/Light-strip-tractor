#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// 装甲板类型常量定义
const int SMALL_ARMOR = 0;
const int BIG_ARMOR = 1;
const int ARMOR_NO = 2;
const float ANGLE_TO_UP = 1.0f;

// 装甲板实际物理尺寸(单位：mm)
const float SMALL_ARMOR_WIDTH = 135.0f;  // 小装甲板宽度
const float SMALL_ARMOR_HEIGHT = 55.0f;  // 小装甲板高度
const float BIG_ARMOR_WIDTH = 230.0f;    // 大装甲板宽度
const float BIG_ARMOR_HEIGHT = 55.0f;    // 大装甲板高度
const float LED_LENGTH = 55.0f;          // LED灯条长度

// 相机参数
const double CAMERA_FX = 1460.0;  // 相机焦距(根据实际相机标定结果修改)
const double CAMERA_FY = 1460.0;
const double CAMERA_CX = 320.0;   // 相机光心(根据实际相机标定结果修改)
const double CAMERA_CY = 240.0;

// 参数结构体定义
struct Params {
    float brightness_threshold;
    float light_min_area;
    float light_max_ratio;
    float light_contour_min_solidity;
    float light_color_detect_extend_ratio;
    float light_max_angle_diff;
    float light_max_height_diff_ratio;
    float light_max_y_diff_ratio;
    float light_min_x_diff_ratio;
    float armor_max_aspect_ratio;
    float armor_min_aspect_ratio;
    float armor_big_armor_ratio;
    float armor_small_armor_ratio;
};

// 灯条描述类
class LightDescriptor {
public:
    Point2f center;
    float length;
    float angle;
    
    LightDescriptor(const RotatedRect& light) {
        center = light.center;
        length = light.size.height;
        angle = light.angle;
    }
};

// 装甲板描述类
class ArmorDescriptor {
public:
    LightDescriptor leftLight;
    LightDescriptor rightLight;
    int type;
    float rotationScore;
    float distance;  // 装甲板距离
    float pitch;     // 俯仰角
    float yaw;       // 偏航角
    
    ArmorDescriptor(const LightDescriptor& left, const LightDescriptor& right, 
                   int armorType, float score, const Params& param) 
        : leftLight(left), rightLight(right), type(armorType), 
          rotationScore(score), distance(0), pitch(0), yaw(0) {}
};

// 全局变量声明
int _flag = ARMOR_NO;
Params _param;
Mat _roiImg;

// 角度调整函数
void adjustRec(RotatedRect& rec, float angle) {
    if(rec.size.width < rec.size.height) {
        swap(rec.size.width, rec.size.height);
        rec.angle += 90.0f;
    }

    if(angle == ANGLE_TO_UP) {
        if(rec.angle >= 45.0) {
            rec.angle = rec.angle - 90.0;
        }
        if(rec.angle < -45.0) {
            rec.angle = 90.0 + rec.angle;
        }
    }
}

/**
 * @brief 预处理图像
 * @param src 输入图像
 * @param dst 输出图像
 * 进行灰度转换、高斯模糊、二值化和膨胀操作
 */
void preprocess(const Mat& src, Mat& dst) {
    // 检查输入图像是否为空
    if(src.empty()) {
        std::cout << "输入图像为空" << std::endl;
        return;
    }
    
    // 确保输入图像是BGR格式
    Mat temp = src.clone();
    if(src.channels() == 1) {
        cvtColor(src, temp, COLOR_GRAY2BGR);
    }
    
    // 转换为灰度图
    cvtColor(temp, dst, COLOR_BGR2GRAY);
    
    // 高斯模糊
    GaussianBlur(dst, dst, Size(5, 5), 0);
    
    // 二值化
    threshold(dst, dst, _param.brightness_threshold, 255, THRESH_BINARY);
    
    // 膨胀操作
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(dst, dst, element);
}

/**
 * @brief 分离颜色通道
 * @param src 输入图像
 * @param grayImg 输出的灰度图像
 * @param isEnemyRed 是否识别红色装甲板
 * 根据敌方颜色分离对应的颜色通道
 */
void splitChannels(Mat& src, Mat& grayImg, bool isEnemyRed) {
    vector<Mat> channels;
    split(src, channels);
    
    if (isEnemyRed) {
        grayImg = channels[2] - channels[0]; // Red minus Blue
    } else {
        grayImg = channels[0] - channels[2]; // Blue minus Red
    }
}

/**
 * @brief 查找可能的灯条轮廓
 * @param src 输入图像
 * @return 灯条轮廓的集合
 */
vector<vector<Point>> findLights(const Mat& src) {
    Mat binBrightImg;
    preprocess(src, binBrightImg);
    
    vector<vector<Point>> lightContours;
    findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return lightContours;
}

/**
 * @brief 检测灯条
 * @param src 输入图像
 * @return 灯条描述符的集合
 */
vector<LightDescriptor> detectLights(const Mat& src) {
    vector<vector<Point>> lightContours = findLights(src);
    vector<LightDescriptor> lightInfos;
    
    for(const auto& contour : lightContours) {
        // 检查轮廓点数是否足够
        if(contour.size() < 5) continue;
        
        float lightContourArea = contourArea(contour);
        if(lightContourArea < _param.light_min_area) continue;

        // 增加try-catch块来处理可能的异常
        try {
            RotatedRect lightRec = fitEllipse(contour);
            adjustRec(lightRec, ANGLE_TO_UP);

            if(lightRec.size.width / lightRec.size.height > _param.light_max_ratio ||
               lightContourArea / lightRec.size.area() < _param.light_contour_min_solidity) {
                continue;
            }

            lightRec.size.width *= _param.light_color_detect_extend_ratio;
            lightRec.size.height *= _param.light_color_detect_extend_ratio;
            
            Rect lightRect = lightRec.boundingRect();
            const Rect srcBound(Point(0, 0), _roiImg.size());
            lightRect &= srcBound;
            
            lightInfos.push_back(LightDescriptor(lightRec));
        }
        catch(const cv::Exception& e) {
            // 忽略拟合失败的轮廓
            continue;
        }
    }

    if(lightInfos.empty()) {
        _flag = ARMOR_NO;
    }
    
    return lightInfos;
}

/**
 * @brief 匹配装甲板
 * @param lightInfos 灯条信息的集合
 * @return 装甲板描述符的集合
 */
vector<ArmorDescriptor> matchArmors(const vector<LightDescriptor>& lightInfos) {
    vector<ArmorDescriptor> armors;
    if (lightInfos.size() < 2) return armors;

    vector<LightDescriptor> lightsSorted = lightInfos;
    sort(lightsSorted.begin(), lightsSorted.end(), 
        [](const LightDescriptor& ld1, const LightDescriptor& ld2) {
            return ld1.center.x < ld2.center.x;
        });

    for (size_t i = 0; i < lightsSorted.size() - 1; i++) {
        for (size_t j = i + 1; j < lightsSorted.size(); j++) {
            const LightDescriptor& leftLight = lightsSorted[i];
            const LightDescriptor& rightLight = lightsSorted[j];

            float angleDiff = abs(leftLight.angle - rightLight.angle);
            float lengthDiffRatio = abs(leftLight.length - rightLight.length) / 
                                   max(leftLight.length, rightLight.length);

            if (angleDiff > _param.light_max_angle_diff ||
                lengthDiffRatio > _param.light_max_height_diff_ratio) {
                continue;
            }

            float distance = cv::norm(leftLight.center - rightLight.center);
            float meanLength = (leftLight.length + rightLight.length) / 2;
            float yDiff = abs(leftLight.center.y - rightLight.center.y);
            float xDiff = abs(leftLight.center.x - rightLight.center.x);
            float yDiffRatio = yDiff / meanLength;
            float xDiffRatio = xDiff / meanLength;
            float distanceRatio = distance / meanLength;

            if (yDiffRatio > _param.light_max_y_diff_ratio ||
                xDiffRatio < _param.light_min_x_diff_ratio ||
                distanceRatio > _param.armor_max_aspect_ratio ||
                distanceRatio < _param.armor_min_aspect_ratio) {
                continue;
            }

            int armorType = (distanceRatio > _param.armor_big_armor_ratio) ? 
                           BIG_ARMOR : SMALL_ARMOR;

            float ratioOffset = (armorType == BIG_ARMOR) ? 
                              max(_param.armor_big_armor_ratio - distanceRatio, 0.0f) :
                              max(_param.armor_small_armor_ratio - distanceRatio, 0.0f);
            float yOffset = yDiff / meanLength;
            float rotationScore = -(ratioOffset * ratioOffset + yOffset * yOffset);

            armors.emplace_back(ArmorDescriptor(leftLight, rightLight, armorType, 
                              rotationScore, _param));
        }
    }

    return armors;
}

/**
 * @brief 计算装甲板位置
 * @param armor 装甲板描述符
 */
void calculateArmorPosition(ArmorDescriptor& armor) {
    // 获取装甲板实际宽度
    float realWidth = (armor.type == BIG_ARMOR) ? BIG_ARMOR_WIDTH : SMALL_ARMOR_WIDTH;
    float realHeight = (armor.type == BIG_ARMOR) ? BIG_ARMOR_HEIGHT : SMALL_ARMOR_HEIGHT;
    
    // 计算装甲板在图像上的宽度
    float pixelWidth = cv::norm(armor.leftLight.center - armor.rightLight.center);
    
    // 使用相似三角形原理计算距离
    armor.distance = (realWidth * CAMERA_FX) / pixelWidth;
    
    // 计算装甲板中心点
    Point2f center = (armor.leftLight.center + armor.rightLight.center) * 0.5f;
    
    // 计算偏航角（水平角度）
    armor.yaw = atan2((center.x - CAMERA_CX) * armor.distance / CAMERA_FX, armor.distance) * 180 / CV_PI;
    
    // 计算俯仰角（垂直角度）
    armor.pitch = atan2((center.y - CAMERA_CY) * armor.distance / CAMERA_FY, armor.distance) * 180 / CV_PI;
}

// 主函数：识别红色装甲板
int main() {
    // 初始化参数
    _param.brightness_threshold = 100;
    _param.light_min_area = 10;
    _param.light_max_ratio = 1.0;
    _param.light_contour_min_solidity = 0.5;
    _param.light_color_detect_extend_ratio = 1.1;
    _param.light_max_angle_diff = 15.0;
    _param.light_max_height_diff_ratio = 0.2;
    _param.light_max_y_diff_ratio = 2.0;
    _param.light_min_x_diff_ratio = 0.5;
    _param.armor_max_aspect_ratio = 5.0;
    _param.armor_min_aspect_ratio = 1.0;
    _param.armor_big_armor_ratio = 3.2;
    _param.armor_small_armor_ratio = 2.0;

    // 打开摄像头或视频文件
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "无法打开摄像头！" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // 存储原始图像
        _roiImg = frame.clone();

        // 分离红色通道
        Mat grayImg;
        splitChannels(frame, grayImg, true);

        // 检测灯条
        vector<LightDescriptor> lights = detectLights(grayImg);

        // 匹配装甲板
        vector<ArmorDescriptor> armors = matchArmors(lights);

        // 绘制检测结果
        for (auto& armor : armors) {
            // 计算装甲板位置
            calculateArmorPosition(armor);
            
            // 绘制装甲板框架
            line(frame, armor.leftLight.center, armor.rightLight.center, Scalar(0, 255, 0), 2);
            circle(frame, armor.leftLight.center, 3, Scalar(0, 0, 255), -1);
            circle(frame, armor.rightLight.center, 3, Scalar(0, 0, 255), -1);
            
            // 显示装甲板信息
            Point textPos = Point((armor.leftLight.center.x + armor.rightLight.center.x)/2,
                                (armor.leftLight.center.y + armor.rightLight.center.y)/2);
            string info = cv::format("D:%.2fm P:%.1f Y:%.1f", 
                                   armor.distance/1000.0, // 转换为米
                                   armor.pitch,
                                   armor.yaw);
            putText(frame, info, textPos, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        // 显示结果
        imshow("装甲板识别", frame);

        // 按ESC退出
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
