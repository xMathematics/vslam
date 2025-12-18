//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"

namespace myslam {

VisualOdometry::VisualOdometry(std::string &config_path)
    : config_file_path_(config_path) {}

// VisualOdometry 类初始化函数，负责整个视觉里程计系统的初始化工作
bool VisualOdometry::Init() {
    // 第一步：读取配置文件
    // 调用 Config 类的静态方法 SetParameterFile，传入配置文件路径
    // 如果读取配置文件失败，返回 false 表示初始化失败
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    // 第二步：创建并初始化数据集对象
    // 从配置文件中获取数据集目录路径，创建 Dataset 智能指针对象
    dataset_ = Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    // 使用 CHECK_EQ 宏检查数据集初始化是否成功（这里假设返回 true 表示成功）
    CHECK_EQ(dataset_->Init(), true);

    // 第三步：创建各个系统组件并建立连接关系
    // 创建前端、后端、地图和可视化器的智能指针对象
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);

    // 第四步：设置前端组件的依赖关系
    // 将后端、地图、可视化器和相机参数设置到前端中
    frontend_->SetBackend(backend_);  // 前端需要与后端通信
    frontend_->SetMap(map_);          // 前端需要更新地图
    frontend_->SetViewer(viewer_);    // 前端可能需要触发可视化更新
    // 从数据集中获取左右相机参数（双目视觉），并设置到前端
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    // 第五步：设置后端组件的依赖关系
    // 将地图和相机参数设置到后端中
    backend_->SetMap(map_);  // 后端需要优化地图
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));  // 后端需要相机参数进行优化

    // 第六步：设置可视化器的依赖关系
    // 将地图设置到可视化器中，用于显示地图点、关键帧等
    viewer_->SetMap(map_);

    // 所有组件初始化完成，返回 true 表示成功
    return true;
}

// 视觉里程计主运行函数，控制整个系统的运行流程
void VisualOdometry::Run() {
    // 进入主循环
    while (1) {
        // 输出运行日志
        LOG(INFO) << "VO is running";
        // 调用 Step() 函数处理单帧数据，如果返回 false 则退出循环
        if (Step() == false) {
            break;
        }
    }

    // 循环结束后，停止后端优化线程（如果后端有独立线程）
    backend_->Stop();
    // 关闭可视化器
    viewer_->Close();

    // 输出退出日志
    LOG(INFO) << "VO exit";
}

// 单步处理函数，负责处理数据集中的每一帧
bool VisualOdometry::Step() {
    // 从数据集中获取下一帧数据
    Frame::Ptr new_frame = dataset_->NextFrame();
    // 如果获取到的帧为空指针，说明数据集已处理完毕，返回 false
    if (new_frame == nullptr) return false;

    // 记录处理开始时间，用于性能统计
    auto t1 = std::chrono::steady_clock::now();
    // 将当前帧添加到前端进行处理，并获取处理是否成功的标志
    bool success = frontend_->AddFrame(new_frame);
    // 记录处理结束时间
    auto t2 = std::chrono::steady_clock::now();
    // 计算处理耗时（转换为秒为单位）
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    // 输出处理耗时日志
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    // 返回处理是否成功的标志
    return success;
}

}  // namespace myslam
