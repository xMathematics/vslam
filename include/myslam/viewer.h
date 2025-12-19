//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

/**
 * 可视化
 */
// 可视化器类，用于显示SLAM系统的运行状态
class Viewer {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 确保Eigen类型的内存对齐
    typedef std::shared_ptr<Viewer> Ptr;  // 定义智能指针类型

    // 构造函数
    Viewer();

    // 设置地图对象
    void SetMap(Map::Ptr map) { map_ = map; }

    // 关闭可视化器
    void Close();

    // 增加一个当前帧用于显示
    void AddCurrentFrame(Frame::Ptr current_frame);

    // 更新地图显示
    void UpdateMap();

   private:
    // 可视化线程的主循环函数
    void ThreadLoop();

    // 绘制一个帧（相机位姿）
    void DrawFrame(Frame::Ptr frame, const float* color);

    // 绘制地图点
    void DrawMapPoints();

    // 让相机跟随当前帧（第一人称视角）
    void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    /// 将当前帧的特征点绘制到图像上
    cv::Mat PlotFrameImage();

    // 当前帧的指针
    Frame::Ptr current_frame_ = nullptr;
    // 地图的指针
    Map::Ptr map_ = nullptr;

    // 可视化线程
    std::thread viewer_thread_;
    // 可视化器运行标志
    bool viewer_running_ = true;

    // 活跃的关键帧集合，用于显示
    std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
    // 活跃的地图点集合，用于显示
    std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
    // 地图更新标志，用于触发重新绘制
    bool map_updated_ = false;

    // 数据互斥锁，保护共享数据
    std::mutex viewer_data_mutex_;
};

}  // namespace myslam

#endif  // MYSLAM_VIEWER_H
