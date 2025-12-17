#pragma once
#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include <opencv2/features2d.hpp>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

class Backend;
class Viewer;

enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

/**
 * 前端
 * 估计当前帧Pose，在满足关键帧条件时向地图加入关键帧并触发优化
 */
/**
 * 视觉SLAM前端处理类
 * 负责特征提取、匹配、位姿估计等核心功能
 */
class Frontend {
public:
    /**
     * 构造函数
     * @param camera 相机模型参数
     * @param config 系统配置参数
     */
    Frontend(Camera::Ptr camera, Config::Ptr config) 
        : camera_(std::move(camera)), 
          config_(std::move(config)) 
    {
        // 初始化ORB特征检测器 (使用OpenCV)
        orb_ = cv::ORB::create(config_->orb_num_features);
        
        // 初始化BFMatcher (汉明距离匹配)
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
        
        // 初始化Eigen矩阵 (用于位姿存储)
        current_pose_ = Eigen::Matrix4d::Identity();  // 当前位姿矩阵
    }

    /**
     * 添加新帧进行处理
     * @param frame 输入帧数据
     * @return 跟踪状态 (SUCCESS/LOST)
     */
    FrontendStatus addFrame(Frame::Ptr frame) {
        current_frame_ = std::move(frame);
        
        switch (status_) {
        case FrontendStatus::INITIALIZING:
            return initialize();  // 系统初始化
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            return track();       // 常规跟踪
        case FrontendStatus::LOST:
            return relocalize();  // 重定位
        }
        return FrontendStatus::LOST;
    }

private:
    /**
     * 系统初始化
     * @return 初始化状态
     */
    FrontendStatus initialize() {
        // 检测ORB特征点 (使用OpenCV)
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb_->detectAndCompute(
            current_frame_->image,          // 输入图像
            cv::noArray(),                  // 掩膜
            keypoints,                      // 输出关键点
            descriptors                     // 输出描述子
        );
        
        // 转换到Eigen格式存储
        current_frame_->features.reserve(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            Eigen::Vector2d position(
                keypoints[i].pt.x, 
                keypoints[i].pt.y
            );
            current_frame_->features.emplace_back(
                position, 
                descriptors.row(i)
            );
        }
        
        // 设置初始关键帧
        insertKeyFrame();
        status_ = FrontendStatus::TRACKING_GOOD;
        return status_;
    }

    /**
     * 帧间跟踪
     * @return 跟踪状态
     */
    FrontendStatus track() {
        // 特征匹配 (使用OpenCV BFMatcher)
        std::vector<cv::DMatch> matches;
        matcher_->match(
            last_frame_->getDescriptors(),   // 上一帧描述子
            current_frame_->getDescriptors(), // 当前帧描述子
            matches                          // 匹配结果
        );
        
        // 筛选优质匹配 (汉明距离阈值)
        std::vector<cv::DMatch> good_matches;
        for (const auto& m : matches) {
            if (m.distance < config_->max_match_distance) {
                good_matches.push_back(m);
            }
        }
        
        // 使用Eigen进行位姿估计 (RANSAC PnP)
        Eigen::Matrix3d R;  // 旋转矩阵
        Eigen::Vector3d t;  // 平移向量
        std::vector<bool> inliers;
        solvePnPRANSAC(
            last_frame_->getPoints3D(),     // 3D点 (Eigen::Vector3d)
            current_frame_->getPoints2D(),   // 2D点 (Eigen::Vector2d)
            good_matches,                   // 匹配对
            R, t, inliers                   // 输出位姿和内点
        );
        
        // 更新当前位姿 (Eigen矩阵运算)
        current_pose_.block<3, 3>(0, 0) = R;
        current_pose_.block<3, 1>(0, 3) = t;
        
        // 判断跟踪质量
        const double inlier_ratio = 
            static_cast<double>(inliers.size()) / good_matches.size();
        status_ = (inlier_ratio > config_->min_inlier_ratio) 
                ? FrontendStatus::TRACKING_GOOD 
                : FrontendStatus::TRACKING_BAD;
        
        return status_;
    }

    /**
     * PnP位姿求解 (使用Eigen)
     * @param pts3d 3D点坐标
     * @param pts2d 2D像素坐标
     * @param matches 特征匹配
     * @param R 输出旋转矩阵
     * @param t 输出平移向量
     * @param inliers 输出内点标记
     */
    void solvePnPRANSAC(
        const std::vector<Eigen::Vector3d>& pts3d,
        const std::vector<Eigen::Vector2d>& pts2d,
        const std::vector<cv::DMatch>& matches,
        Eigen::Matrix3d& R,
        Eigen::Vector3d& t,
        std::vector<bool>& inliers
    ) {
        // 将Eigen向量转换为OpenCV格式
        std::vector<cv::Point3f> cv_pts3d;
        std::vector<cv::Point2f> cv_pts2d;
        for (const auto& m : matches) {
            cv_pts3d.push_back(toCVPoint3f(pts3d[m.queryIdx]));
            cv_pts2d.push_back(toCVPoint2f(pts2d[m.trainIdx]));
        }
        
        // 使用OpenCV求解PnP
        cv::Mat rvec, tvec, inliers_cv;
        cv::solvePnPRansac(
            cv_pts3d, cv_pts2d,
            camera_->getCVMatrix(),  // 相机内参
            cv::Mat(),               // 畸变系数
            rvec, tvec, 
            false,                   // 不使用初始猜测
            config_->pnp_iters,      // RANSAC迭代次数
            config_->pnp_reproj_th,  // 重投影阈值
            config_->pnp_confidence, // 置信度
            inliers_cv               // 内点索引
        );
        
        // 转换回Eigen格式
        cv::Rodrigues(rvec, R_cv);  // 旋转向量转旋转矩阵
        R = toEigenMatrix3d(R_cv);
        t = toEigenVector3d(tvec);
        
        // 生成内点标记
        inliers.resize(matches.size(), false);
        for (int i = 0; i < inliers_cv.rows; ++i) {
            inliers[inliers_cv.at<int>(i)] = true;
        }
    }

    // 成员变量
    Camera::Ptr camera_;           // 相机模型
    Config::Ptr config_;           // 配置参数
    Frame::Ptr current_frame_;     // 当前帧
    Frame::Ptr last_frame_;        // 上一帧
    Eigen::Matrix4d current_pose_; // 当前位姿 (4x4齐次矩阵)
    FrontendStatus status_ = FrontendStatus::INITIALIZING;  // 状态机
    
    // OpenCV对象
    cv::Ptr<cv::ORB> orb_;         // ORB特征检测器
    cv::Ptr<cv::DescriptorMatcher> matcher_;  // 特征匹配器
};


}  // namespace myslam

#endif  // MYSLAM_FRONTEND_H
