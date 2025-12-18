#pragma once

#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"

namespace myslam {

// 针孔双目相机模型
class Camera {
   public:
    // Eigen内存对齐宏，确保使用Eigen库的类在STL容器中能正确内存对齐
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    // 定义智能指针类型别名，方便使用
    typedef std::shared_ptr<Camera> Ptr;

    // 相机内参：焦距(fx, fy)，主点(cx, cy)，和双目基线距离
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0,
           baseline_ = 0;  // 相机内参
           
    // 外参：从双目相机坐标系到单个相机坐标系的变换
    // 通常用于表示左右相机之间的相对位姿
    SE3 pose_;             // 外参，从双目相机到单个相机的变换
    SE3 pose_inv_;         // 外参的逆变换，预计算以提高效率

    // 默认构造函数
    Camera();

    // 带参数的构造函数，初始化所有内参和外参
    Camera(double fx, double fy, double cx, double cy, double baseline,
           const SE3 &pose)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
        // 计算并存储外参的逆变换
        pose_inv_ = pose_.inverse();
    }

    // 获取外参变换
    SE3 pose() const { return pose_; }

    // 返回相机内参矩阵K
    Mat33 K() const {
        Mat33 k;  // 创建3x3矩阵
        // 设置内参矩阵（注意：这里假设没有畸变）
        k << fx_, 0, cx_,  // 第一行：fx, 0, cx
             0, fy_, cy_,  // 第二行：0, fy, cy
             0, 0, 1;      // 第三行：0, 0, 1
        return k;
    }

    // 坐标变换函数声明：

    // 世界坐标系 → 相机坐标系
    // p_w: 世界坐标系下的3D点
    // T_c_w: 世界到相机的变换矩阵
    Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

    // 相机坐标系 → 世界坐标系
    // p_c: 相机坐标系下的3D点
    // T_c_w: 世界到相机的变换矩阵
    Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

    // 相机坐标系 → 像素坐标系（投影）
    // p_c: 相机坐标系下的3D点
    // 返回：像素坐标(u,v)
    Vec2 camera2pixel(const Vec3 &p_c);

    // 像素坐标系 → 相机坐标系（反投影）
    // p_p: 像素坐标(u,v)
    // depth: 深度值（默认为1）
    // 返回：相机坐标系下的3D点
    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

    // 像素坐标系 → 世界坐标系
    // p_p: 像素坐标(u,v)
    // T_c_w: 世界到相机的变换矩阵
    // depth: 深度值
    Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1);

    // 世界坐标系 → 像素坐标系
    // p_w: 世界坐标系下的3D点
    // T_c_w: 世界到相机的变换矩阵
    Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);
};

}  // namespace myslam
#endif  // MYSLAM_CAMERA_H
