#pragma once
#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "myslam/backend.h"
#include "myslam/common_include.h"
#include "myslam/dataset.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"

namespace myslam {

/**
 * VO 对外接口
 */
class VisualOdometry {
public:
    // 确保Eigen库中的数据结构内存对齐，避免段错误
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    // 定义智能指针别名，方便使用智能指针管理该类的实例
    typedef std::shared_ptr<VisualOdometry> Ptr;

    /// 构造函数：传入配置文件路径
    VisualOdometry(std::string &config_path);

    /**
     * 在运行VO之前进行初始化操作
     * @return 初始化成功返回true，否则false
     */
    bool Init();

    /**
     * 在数据集上启动视觉里程计（VO）
     */
    void Run();

    /**
     * 在数据集中向前推进一帧（单步执行）
     */
    bool Step();

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

private:
    // 标记VO系统是否已初始化
    bool inited_ = false;
    
    // 配置文件的路径
    std::string config_file_path_;

    // 前端处理模块的智能指针
    Frontend::Ptr frontend_ = nullptr;
    
    // 后端优化模块的智能指针
    Backend::Ptr backend_ = nullptr;
    
    // 地图管理模块的智能指针
    Map::Ptr map_ = nullptr;
    
    // 可视化模块的智能指针
    Viewer::Ptr viewer_ = nullptr;

    // 数据集模块的智能指针，用于读取和提供图像数据
    Dataset::Ptr dataset_ = nullptr;
};
}  // namespace myslam

#endif  // MYSLAM_VISUAL_ODOMETRY_H
