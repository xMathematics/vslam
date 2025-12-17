#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam {

struct Frame;

struct Feature;

/**
 * 路标点类
 * 特征点在三角化之后形成路标点
 */
// 定义地图点结构体，表示三维空间中的特征点
struct MapPoint {
    // 确保Eigen对象内存对齐（防止SSE指令访问错误地址）
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 类型别名：MapPoint的智能指针类型
    typedef std::shared_ptr<MapPoint> Ptr;

    // 成员变量：地图点的唯一ID（用于标识不同地图点）
    unsigned long id_;

    // 标志位：是否为异常点（true表示该点可能由误匹配产生，需剔除）
    bool is_outlier_ = false;

    // 地图点在世界坐标系中的3D坐标（使用Eigen::Vector3d类型）
    Vec3 pos_ = Vec3::Zero();

    // 观测方向向量（从地图点指向相机的归一化方向）
    Vec3 norm_ = Vec3::Zero();

    // 描述子（用于特征匹配的二进制描述子）
    cv::Mat descriptor_;

    // 被观测次数（统计该点在多少帧中被观测到）
    int observed_times_ = 0;

    // 构造函数：初始化ID和3D位置
    MapPoint(long id, const Vec3 &position) 
        : id_(id), pos_(position) {}

    // 工厂方法：创建新地图点
    static MapPoint::Ptr CreateNewMappoint() {
        static long factory_id = 0;  // 静态ID生成器
        MapPoint::Ptr new_mappoint(new MapPoint(factory_id++, Vec3::Zero()));
        return new_mappoint;
    }

    // 设置地图点的3D位置
    void SetPos(const Vec3 &pos) { 
        pos_ = pos; 
    }

    // 获取地图点的3D位置
    Vec3 GetPos() const { 
        return pos_; 
    }
};

}  // namespace myslam

#endif  // MYSLAM_MAPPOINT_H
