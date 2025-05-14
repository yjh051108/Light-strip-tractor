#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class ArmorDetector {
private:
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
        float min_aspect_ratio = 2.5f;
        float max_aspect_ratio = 100.0f;
    };

    struct LightBar {
        cv::RotatedRect rect;
        cv::Point2f top_point, bottom_point;
        float length, brightness;
        cv::Vec3b color;

        LightBar();
        LightBar(const cv::RotatedRect& r, float b, const cv::Scalar& c);
        LightBar(cv::Point2f p1, cv::Point2f p2);
    };

    struct ArmorPair {
        LightBar left, right;
        ArmorPair(const LightBar& l={}, const LightBar& r={});
    };

    DetectionParams params;
    cv::Mat all_rect_debug;
    cv::Mat debug_img;

    bool isSimilarPair(const ArmorPair& p1, const ArmorPair& p2, float threshold = 10.0f);
    cv::Mat getBrightnessFilteredImage(const cv::Mat& gray);
    cv::Mat getAreaFilteredImage(const cv::Mat& binary);
    std::vector<ArmorPair> matchLightBars(const std::vector<LightBar>& lights);
    void showDebugInfo(const cv::Mat& frame, const std::vector<LightBar>& lights);

public:
    std::vector<cv::Point2f> detectArmorCenters(const cv::Mat& frame);
};